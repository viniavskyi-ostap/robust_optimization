import torch


def weighted_least_squares_mean_fit(X, weights=None):
    """
    Solves weighted least squares problem for mean fitting.
    Args:
        X: tensor of input points (n, d)
        weights: tensor of weight for each point's residual (n,)

    Returns:
        solution of weighted least squares (k,) and corresponding residuals
    """
    if weights is None:
        weights = torch.ones(X.shape[0], device=X.device)
    theta = (X * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
    residuals = torch.linalg.norm(X - theta, dim=-1)
    return theta, residuals


def weighted_least_squares_mean_fit_batched(X, weights=None):
    """
    Solves weighted least squares problem for mean fitting.
    Args:
        X: tensor of input points (b, n, d)
        weights: tensor of weight for each point's residual (b, n)

    Returns:
        solution of weighted least squares (b, k) and corresponding residuals
    """
    b, n, d = X.size()
    if weights is None:
        weights = torch.ones(b, n, device=X.device)
    weights = weights.unsqueeze(-1)
    theta = (X * weights).sum(dim=1) / weights.sum(dim=1)
    vector_residuals = X - theta.unsqueeze(1)
    residuals = torch.linalg.norm(vector_residuals, dim=-1)
    return theta, residuals, vector_residuals, weights


def irls(X, kernel, scale=1., init_theta=None, init_residuals=None, min_iter=0, max_iter=5, eta=0.2,
         return_loss_history=False, return_thetas_history=False):
    """
    Iteratively-reweighted least squares for robust mean estimation
    Args:
        X: array of input points (n, k)
        kernel: kernel function used to calculate weights
        scale: scale of the kernel to apply
        init_theta: initial value of parameters to be estimated
        init_residuals: initial value of residuals for corresponding init_theta
        min_iter: minimum number of iterations to perform
        max_iter: maximum number of iterations to perform
        eta: relative stopping criterion threshold
        return_loss_history: return loss function history flag
        return_thetas_history: return evolution of thetas flag
    Returns:
        np.array: robust estimation of the mean in the input point set
        np.array: residuals after the final estimation stage
    """
    if max_iter < min_iter:
        max_iter = min_iter

    if init_theta is not None:
        if init_residuals is None:
            raise ValueError('init_residuals must be provided when init_theta is not None')
        theta, previous_residuals = init_theta, init_residuals
    else:
        # initial guess based on standard LS
        theta, previous_residuals = weighted_least_squares_mean_fit(X)
    previous_loss = kernel(previous_residuals, scale)
    residuals, loss = previous_residuals, previous_loss

    if return_loss_history:
        loss_history = []
    if return_thetas_history:
        theta_history = []

    zero_scalar = torch.tensor(0., device=X.device)

    for i in range(max_iter):
        weights = kernel.get_weight(previous_residuals, scale)
        # check if sum of weights is positive
        if torch.isclose(weights.sum(), zero_scalar):
            break
        theta, residuals = weighted_least_squares_mean_fit(X, weights)
        loss = kernel(residuals, scale)

        if return_loss_history:
            loss_history.append(loss.detach().mean())
        if return_thetas_history:
            theta_history.append(theta.detach())

        # don't check stopping criterion before min_iter
        if i >= min_iter:
            idx = residuals > previous_residuals

            kernel_residual_diff = loss - previous_loss
            delta_increased = (kernel_residual_diff[idx]).sum()
            delta_decreased = -(kernel_residual_diff[~idx]).sum()

            if torch.isclose(delta_decreased, zero_scalar) and torch.isclose(delta_decreased, zero_scalar):
                break

            stopping_criterion = (delta_decreased - delta_increased) / (delta_decreased + delta_increased)
            if stopping_criterion <= eta:
                break
        previous_residuals = residuals
        previous_loss = loss

    return_values = [theta, residuals]
    if return_loss_history:
        return_values.append(torch.stack(loss_history))
    if return_thetas_history:
        return_values.append(torch.stack(theta_history))
    return return_values


if __name__ == '__main__':
    X = torch.randn(2, 5, 3)
    print(weighted_least_squares_mean_fit_batched(X)[-1].shape)
