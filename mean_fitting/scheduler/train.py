import os
import yaml

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from mean_fitting.scheduler.data.datamodule import Noisy2DPointsDataModule
from mean_fitting.scheduler.go_scale_regressor_v0 import GraduatedOptimizationRegressor

if __name__ == '__main__':
    pl.seed_everything(0)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')) as f:
        config = yaml.full_load(f)

    dm = Noisy2DPointsDataModule(**config['data'])
    # dm.setup()
    # dl = dm.train_dataloader()
    # X, y = next(iter(dl))
    # print(X.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(X[0, :, 0], X[0, :, 1])
    # plt.show()
    # exit(0)

    model = GraduatedOptimizationRegressor(point_dim=len(config['data']['low']), **config['model'])

    trainer = pl.Trainer(log_every_n_steps=1, gpus=[0], accelerator='dp',
                         plugins=DDPPlugin(find_unused_parameters=False),
                         max_epochs=5)
    trainer.fit(model, datamodule=dm)
