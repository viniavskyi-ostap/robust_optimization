import numpy as np


def noisy_point_cloud(mean, cov, num_points, outliers_ratio, outliers_low, outliers_high):
    """
    Generate noisy n-dim point cloud.
    Inliers are generated according to gaussian process with given mean and covariance matrix.
    Outliers are sampled uniformly in predefined range
    Args:
        mean: mean of inliers distribution
        cov: covariance of inliers distribution
        num_points: number of points in the resulting point cloud
        outliers_ratio: fraction of outliers points in the point cloud
        outliers_low: lower bound on outliers sampling process
        outliers_high: upper bound on outliers sampling process
    Returns:
        n-dim point cloud containing inliers and outliers
    """
    num_outliers = int(num_points * outliers_ratio)
    num_inliers = num_points - num_outliers

    print(mean.shape, cov.shape)
    inliers = np.random.multivariate_normal(mean, cov, size=num_inliers)
    outliers = np.random.uniform(outliers_low, outliers_high, size=(num_outliers, inliers.shape[-1]))

    return np.concatenate([inliers, outliers], axis=0)
