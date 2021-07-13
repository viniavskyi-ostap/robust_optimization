"""
This module implements a GO system that predicts a scale out of 10 predefined only at the first step.
Then standard rule-based scheduling is run for the next few iterations. For each of 10 possibilities we calculate
the distance of estimated parameters vector to the g.t. Then those distances are converted to a distribution, which
are considered a g.t.
"""
import os
import yaml
from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn as nn

from mean_fitting.irls import irls
from mean_fitting.kernels import WelschKernel
from mean_fitting.scheduler.data.datamodule import Noisy2DPointsDataModule


class GO_Classifier_v0(pl.LightningModule):
    def __init__(self):
        super(GO_Classifier_v0, self).__init__()

        self.num_irls_steps_before_prediction = 3
        self.choices = [0.8, 0.2]
        self.initial_scale = 1.0
        self.num_go_steps_after_prediction = 3
        self.rule_based_scale_multiplier = 0.5
        self.softmax_temperature = 5

        self.lr = 0.001

        self.model = nn.Sequential(
            nn.Linear(self.num_irls_steps_before_prediction, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, len(self.choices))
        )

        self.kernel = WelschKernel()

    def _run_graduated_optimization(self, X):
        """
        Args:
            X: batch of input points (b, n, d)

        Returns:
            list of theta estimates at each step of graduated optimization
        """
        b, n, d = X.size()
        device = X.device

        # run for 10 steps irls on initial scale
        theta, residuals = torch.empty(b, d, device=device), torch.empty(b, n, device=device)
        loss_histories = torch.empty(b, self.num_irls_steps_before_prediction, device=device)

        for i in range(b):
            theta[i], residuals[i], loss_histories[i] = irls(X[i], self.kernel, self.initial_scale,
                                                             min_iter=self.num_irls_steps_before_prediction,
                                                             max_iter=self.num_irls_steps_before_prediction,
                                                             return_loss_history=True)
        # predict first scale change using NN
        choices_logits = self.model(loss_histories)

        thetas = []  # final theta estimates for different choices will be stored here

        for scale in self.choices:
            new_theta, new_residuals = theta.clone(), residuals.clone()
            for i in range(b):
                for j in range(self.num_go_steps_after_prediction):  # run 2 more g.o. steps
                    new_theta[i], new_residuals[i] = irls(X[i], self.kernel,
                                                          scale * (self.rule_based_scale_multiplier ** j),
                                                          init_theta=new_theta[i],
                                                          init_residuals=new_residuals[i])
            thetas.append(new_theta)
        return thetas, choices_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        thetas, choices_logits = self._run_graduated_optimization(x)
        proximity = torch.stack([(y - theta).pow(2).sum(-1) for theta in thetas], dim=-1)
        dist_true = torch.softmax(-proximity * self.softmax_temperature, dim=-1)
        print(proximity)
        print(dist_true)
        log_dist_pred = torch.log_softmax(choices_logits, dim=-1)

        ce_loss = - (dist_true * log_dist_pred).sum(-1).mean()
        self.log('train CE loss', ce_loss.detach(), prog_bar=True, sync_dist=True)

        return ce_loss

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


def train():
    pl.seed_everything(0)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/go_classifier_v0.yaml')) as f:
        config = yaml.full_load(f)

    dm = Noisy2DPointsDataModule(**config['data'])

    model = GO_Classifier_v0()

    trainer = pl.Trainer(log_every_n_steps=1, gpus=[0], accelerator='dp',
                         max_epochs=5)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    # train()
    pl.seed_everything(1)
    model = GO_Classifier_v0()
    X = torch.randn(3, 5, 2)
    y = torch.randn(3, 2)
    model.training_step((X, y), 0)
