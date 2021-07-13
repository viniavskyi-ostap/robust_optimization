"""
In this module is implemented a learnable graduated optimization algorithm.
The next scale that algorithm selects is predicted based on previous scale, point set and current vector residuals set.
The scale is regressed using PointNet architecture.
The loss function is computed based on the proximity of estimated parameter vector and the ground truth one:
Loss = sum_i[ L_2 (theta_gt, theta_estimated_i) ]
The gradients are backpropagated trough IRLS steps.

#TODO: predicts worse than rule based
"""

from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from mean_fitting.irls import weighted_least_squares_mean_fit, irls
from mean_fitting.kernels import WelschKernel
from mean_fitting.scheduler.pointnet import PointNetClassifier


class GraduatedOptimizationRegressor(pl.LightningModule):
    def __init__(self, point_dim=2, num_scales=5, min_scale_multiplier=0.2, max_scale_multiplier=0.8,
                 feat_trans_loss_weight=0.01, lr=0.0001):
        super(GraduatedOptimizationRegressor, self).__init__()

        self.model = PointNetClassifier(output_dim=1, input_dim=(2 * point_dim + 1), point_dim=point_dim)
        self.kernel = WelschKernel()

        self.num_scales = num_scales
        self.min_scale_multiplier = min_scale_multiplier
        self.max_scale_multiplier = max_scale_multiplier
        self.feat_trans_loss_weight = feat_trans_loss_weight
        self.lr = lr

    def forward(self, X):
        return self._run_step(X)[0]

    def _run_graduated_optimization(self, X):
        """
        Args:
            X: batch of input points (b, n, d)

        Returns:
            list of theta estimates at each step of graduated optimization
        """
        b, n, d = X.size()
        device = X.device
        # make initial estimate by ordinary LS
        theta, residuals = torch.empty(b, d, device=device), torch.empty(b, n, device=device)
        for i in range(b):
            theta[i], residuals[i] = weighted_least_squares_mean_fit(X[i])
        scale = residuals.max(dim=1)[0]

        thetas = []
        feat_transformations = []

        for _ in range(self.num_scales):
            scale, feat_trans = self._predict_scale(X, theta, scale)
            new_theta, new_residuals = torch.empty(b, d, device=device), torch.empty(b, n, device=device)

            for i in range(b):
                new_theta[i], new_residuals[i] = irls(X[i], self.kernel, scale[i], theta[i], residuals[i], max_iter=10)

            theta, residuals = new_theta, new_residuals
            thetas.append(theta)
            feat_transformations.append(feat_trans)
        return thetas, feat_transformations

    def _predict_scale(self, X, theta, scale_prev):
        theta = theta.detach()
        scale_prev = scale_prev.detach()
        residuals = X - theta[:, None]
        input = torch.cat([X, residuals, scale_prev[:, None, None].expand(-1, X.size(1), -1)], dim=2)

        input = input.transpose(2, 1)
        scale_multiplier, feat_trans = self.model(input)
        scale_multiplier = torch.sigmoid(scale_multiplier[:, 0]) * (
                self.max_scale_multiplier - self.min_scale_multiplier) + self.min_scale_multiplier
        return scale_multiplier * scale_prev, feat_trans

    def _run_step(self, X):
        thetas, feat_transformations = self._run_graduated_optimization(X)
        return thetas, feat_transformations

    @staticmethod
    def _criterion(y, thetas, feat_transformations):
        # loss on theta
        loss_params = sum(F.mse_loss(theta, y) for theta in thetas)
        # loss on feat_transformations
        loss_trans = sum(
            GraduatedOptimizationRegressor.feature_transform_regularization(feat_trans) for feat_trans in
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
        thetas, feat_transformations = self._run_step(x)
        loss_params, loss_trans = self._criterion(y, thetas, feat_transformations)
        self.log('train loss params', loss_params.detach(), prog_bar=True, sync_dist=True)
        self.log('train loss trans', loss_trans.detach(), prog_bar=True, sync_dist=True)
        return loss_params + self.feat_trans_loss_weight * loss_trans
