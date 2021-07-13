from mean_fitting.irls import irls, weighted_least_squares_mean_fit


def graduated_optimization(X, kernel, go_steps=5):
    """
    Args:

    Returns:

    """
    theta, residuals = weighted_least_squares_mean_fit(X)
    scale = residuals.max(0)[0]

    theta_go_history = []
    scales_history = []

    for _ in range(go_steps):
        theta, residuals, theta_history = irls(
            X, kernel, scale=scale, init_theta=theta, init_residuals=residuals,
            return_thetas_history=True
        )

        scales_history.append(scale.item())
        theta_go_history.append(theta_history)

        scale = scale / 2
    return theta, theta_go_history, scales_history
