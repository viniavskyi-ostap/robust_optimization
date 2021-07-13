import pytorch_lightning as pl
import torch
import numpy as np

from .dataset import Noisy2DPointsDataset


class Noisy2DPointsDataModule(pl.LightningDataModule):
    def __init__(self, low, high, num_points_range, batch_size, outliers_ratio_range, cov_eigenvalues_range,
                 ds_size, num_workers):
        super(Noisy2DPointsDataModule, self).__init__()
        self.low = low
        self.high = high
        self.num_points_range = num_points_range
        self.batch_size = batch_size
        self.outliers_ratio_range = outliers_ratio_range
        self.cov_eigenvalues_range = cov_eigenvalues_range
        self.ds_size = ds_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = Noisy2DPointsDataset(
            low=np.array(self.low),
            high=np.array(self.high),
            num_points_range=self.num_points_range,
            batch_size=self.batch_size,
            outliers_ratio_range=self.outliers_ratio_range,
            cov_eigenvalues_range=self.cov_eigenvalues_range,
            ds_size=self.ds_size
        )
        # self.val_ds = Noisy2DPointsDataset(
        #     low=np.array([-1.0, -1.0]),
        #     high=np.array([1.0, 1.0]),
        #     num_points_range=(100, 1000),
        #     batch_size=self.batch_size,
        #     outliers_ratio_range=(0.4, 0.6),
        #     ds_size=100
        # )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=None,
            num_workers=self.num_workers
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.val_ds,
    #         batch_size=None,
    #         num_workers=self.num_workers
    #     )
