"""
In this module is implemented a learnable graduated optimization algorithm.
The next scale that algorithm selects is predicted based on previous scale, point set and current vector residuals set.
The scale is regressed using PointNet architecture.
The loss function is computed based on the proximity of estimated parameter vector and the ground truth one:
Loss = sum_i[ L_2 (theta_gt, theta_estimated_i) ]
The gradients are backpropagated trough IRLS steps.

"""
from itertools import chain

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from mean_fitting.irls import weighted_least_squares_mean_fit_batched
from mean_fitting.scheduler.pointnet import PointNetSegmentation


class IRLSWeightRegressor(pl.LightningModule):
    def __init__(self, point_dim=2, num_iter=5,
                 feat_trans_loss_weight=0.01, lr=0.0001):
        super(IRLSWeightRegressor, self).__init__()

        self.model = PointNetSegmentation(output_dim=1, input_dim=(2 * point_dim + 1), point_dim=point_dim)

        self.num_iter = num_iter
        self.feat_trans_loss_weight = feat_trans_loss_weight
        self.lr = lr

    def forward(self, X):
        return self._run_irls(X)[0]

    def _run_irls(self, X):
        """
        Args:
            X: batch of input points (b, n, d)

        Returns:
            list of theta estimates at each step of graduated optimization
        """
        # make initial estimate by ordinary LS
        theta, residuals, vector_residuals, weights_prev = weighted_least_squares_mean_fit_batched(X)

        thetas = []
        feat_transformations = []

        for _ in range(self.num_iter):
            weights, feat_trans = self._predict_weights(X, vector_residuals, weights_prev)
            theta, residuals, vector_residuals, weights_prev = weighted_least_squares_mean_fit_batched(X, weights)

            thetas.append(theta)
            feat_transformations.append(feat_trans)
        return thetas, feat_transformations

    def _predict_weights(self, X, vector_residuals, weights_prev):
        vector_residuals = vector_residuals.detach()
        weights_prev = weights_prev.detach()
        input = torch.cat([X, vector_residuals, weights_prev], dim=2)
        input = input.transpose(2, 1)

        weights, feat_trans = self.model(input)
        weights = torch.sigmoid(weights)
        return weights.squeeze(1), feat_trans

    @staticmethod
    def _criterion(y, thetas, feat_transformations):
        # loss on theta
        loss_params = sum(F.mse_loss(theta, y) for theta in thetas)
        # loss on feat_transformations
        loss_trans = sum(
            IRLSWeightRegressor.feature_transform_regularization(feat_trans) for feat_trans in
            feat_transformations)
        return loss_params, loss_trans

    @staticmethod
    def feature_transform_regularization(trans):
        d = trans.size()[1]
        I = torch.eye(d, device=trans.device).unsqueeze(0)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.model.parameters()),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=0.9995
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        thetas, feat_transformations = self._run_irls(x)
        loss_params, loss_trans = self._criterion(y, thetas, feat_transformations)
        self.log('train loss params', loss_params.detach(), prog_bar=True, sync_dist=True)
        self.log('train loss trans', loss_trans.detach(), prog_bar=True, sync_dist=True)
        return loss_params + self.feat_trans_loss_weight * loss_trans


def train():
    import os
    import yaml

    from mean_fitting.scheduler.data.datamodule import Noisy2DPointsDataModule

    pl.seed_everything(0)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/irls_weight_regressor_v0.yaml')) as f:
        config = yaml.full_load(f)

    dm = Noisy2DPointsDataModule(**config['data'])

    model = IRLSWeightRegressor(point_dim=len(config['data']['low']), **config['model'])

    trainer = pl.Trainer(log_every_n_steps=1, gpus=[0], accelerator='dp',
                         max_epochs=15)
    trainer.fit(model, datamodule=dm)


def run():
    import tqdm
    import matplotlib.pyplot as plt
    from mean_fitting.scheduler.data.dataset import Noisy2DPointsDataset
    from mean_fitting.kernels import WelschKernel
    from mean_fitting.graduated_optimization import graduated_optimization
    from mean_fitting.utils.vis import plot_loss_on_scales

    pl.seed_everything(0)

    module = IRLSWeightRegressor.load_from_checkpoint(
        '../../lightning_logs/version_17/checkpoints/epoch=14-step=14999.ckpt')
    module.num_iter = 10
    module.eval()
    ds = Noisy2DPointsDataset(
        low=np.array([-1.0] * 2),
        high=np.array([1.0] * 2),
        num_points_range=(1000, 1001),
        batch_size=1,
        outliers_ratio_range=(0.8, 0.8),
        cov_eigenvalues_range=(0.05, 0.05),
        ds_size=100
    )

    losses_learned = []
    loss_rule_based = []

    for i in tqdm.tqdm(range(len(ds))):
        X, y = ds[i]
        X = torch.from_numpy(X)

        with torch.no_grad():
            thetas = module(X)
            theta = thetas[-1][0].numpy()
        thetas = torch.cat(thetas, dim=0).numpy()
        loss_history = np.linalg.norm(thetas - y, axis=-1)
        # plt.plot(np.arange(loss_history.shape[0]), loss_history)

        losses_learned.append(np.linalg.norm(theta - y))

        theta, theta_go_history, scales_history = graduated_optimization(X[0], kernel=WelschKernel())
        loss_rule_based.append(np.linalg.norm(theta - y))

        # num_iter_per_scale = [len(th) for th in theta_go_history]
        loss_history = np.linalg.norm(torch.cat(theta_go_history, dim=0).numpy() - y, axis=-1)
        # plt.plot(np.arange(loss_history.shape[0]), loss_history)
        # plt.legend(['Learned', 'Rule-based'])
        # plt.show()

    losses_learned = np.array(losses_learned)
    # losses_learned = losses_learned[~np.isnan(losses_learned)]

    loss_rule_based = np.array(loss_rule_based)
    # loss_rule_based = loss_rule_based[~np.isnan(loss_rule_based)]
    fig, axes = plt.subplots(ncols=2)
    axes[0].hist(losses_learned)
    axes[1].hist(loss_rule_based)
    plt.show()
    print(np.mean(losses_learned))
    print(np.mean(loss_rule_based))


if __name__ == '__main__':
    # train()
    run()
