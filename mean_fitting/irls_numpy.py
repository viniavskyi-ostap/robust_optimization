import numpy as np


def weighted_least_squares_mean_fit(X, weights=None):
    """
    Solves weighted least squares problem for mean fitting.
    Args:
        X: array of input points (n, k)
        weights: array of weight for each point's residual (n,)

    Returns:
        solution of weighted least squares (k,) and corresponding residuals
    """
    if weights is None:
        weights = np.ones((X.shape[0],))
    theta = (X * weights[..., np.newaxis]).sum(axis=0) / weights.sum()
    residuals = np.linalg.norm(X - theta, axis=-1)
    return theta, residuals


def irls(X, kernel, init_theta=None, init_residuals=None, max_iter=100, eta=0.2):
    """
    Iteratively-reweighted least squares for robust mean estimation
    Args:
        X: array of input points (n, k)
        kernel: kernel function used to calculate weights
        init_theta: initial value of parameters to be estimated
        init_residuals: initial value of residuals for corresponding init_theta
        max_iter: maximum number of iteration to perform
        eta: relative stopping criterion threshold
    Returns:
        np.array: robust estimation of the mean in the input point set
        np.array: residuals after the final estimation stage
    """
    if init_theta is not None:
        if init_residuals is None:
            raise ValueError('init_residuals must be provided when init_theta is not None')
        theta, previous_residuals = init_theta, init_residuals
    else:
        # initial guess based on standard LS
        theta, previous_residuals = weighted_least_squares_mean_fit(X)

    for i in range(max_iter):
        weights = kernel.get_weight(previous_residuals)
        theta, residuals = weighted_least_squares_mean_fit(X, weights)

        # stopping criterion
        idx = residuals > previous_residuals

        delta_increased = np.sum(kernel(residuals[idx]) - kernel(previous_residuals[idx]))
        delta_decreased = np.sum(kernel(previous_residuals[~idx]) - kernel(residuals[~idx]))

        if np.isclose(delta_decreased, 0.) and np.isclose(delta_decreased, 0.):
            break

        stopping_criterion = (delta_decreased - delta_increased) / (delta_decreased + delta_increased)
        previous_residuals = residuals

        if stopping_criterion <= eta:
            break

    return theta, previous_residuals
