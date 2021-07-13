import numpy as np
import torch


def sample_positive_semidefinite_2d_matrix(lambda1, lambda2):
    # sample direction of first eigenvector
    theta = 2 * np.pi * np.random.random()
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    a = lambda2 + cos_theta ** 2 * (lambda1 - lambda2)
    d = lambda1 + cos_theta ** 2 * (lambda2 - lambda1)
    b = (lambda1 - lambda2) * sin_theta * cos_theta
    return np.array([
        [a, b],
        [b, d]
    ])


def sample_positive_semidefinite_nd_matrix(lambdas):
    n = lambdas.shape[0]
    # sample first eigenvector with non-zero  first component
    v0 = None
    while v0 is None or np.isclose(v0[0], 0.):
        v0 = np.random.uniform(-1, 1, size=(n,))
    # make matrix with v0 and (n-1) mutually independent vectors
    a = np.eye(n)
    a[:, 0] = v0
    # make qr decomposition to obtain orthogonal basis
    q, _ = np.linalg.qr(a, mode='complete')
    return q @ np.diag(lambdas) @ q.T


class Noisy2DPointsDataset(torch.utils.data.Dataset):
    def __init__(self, low, high, num_points_range, batch_size, outliers_ratio_range, cov_eigenvalues_range, ds_size):
        self.low = low
        self.high = high
        self.num_points_low, self.num_points_high = num_points_range
        self.batch_size = batch_size
        self.outliers_ratio_low, self.outliers_ratio_high = outliers_ratio_range
        self.cov_eigenvalues_low, self.cov_eigenvalues_high = cov_eigenvalues_range
        self.size = ds_size
        self.d = low.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        num_points = np.random.randint(self.num_points_low, self.num_points_high)

        X, y = [], []

        for _ in range(self.batch_size):
            outliers_ratio = np.random.uniform(self.outliers_ratio_low, self.outliers_ratio_high)
            num_outliers = int(num_points * outliers_ratio)
            num_inliers = num_points - num_outliers

            mean = np.random.uniform(self.low, self.high)
            lambdas = np.random.uniform(self.cov_eigenvalues_low, self.cov_eigenvalues_high,  size=(self.d,))
            cov = sample_positive_semidefinite_nd_matrix(lambdas)
            inliers = np.random.multivariate_normal(mean, cov, size=num_inliers)
            outliers = np.random.uniform(self.low, self.high, size=(num_outliers, self.d))

            X.append(np.concatenate([inliers, outliers], axis=0))
            y.append(mean)

        return np.stack(X, axis=0).astype(np.float32), np.stack(y, axis=0).astype(np.float32)


if __name__ == '__main__':
    a = sample_positive_semidefinite_nd_matrix(np.array([1.0, 2.0]))
    print(a)
    print(np.linalg.eigvals(a))
