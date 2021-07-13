import numpy as np
import torch
import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from mean_fitting.graduated_optimization import graduated_optimization
from mean_fitting.kernels import WelschKernel
from mean_fitting.scheduler.data.dataset import Noisy2DPointsDataset
from mean_fitting.scheduler.go_scale_regressor_v0 import GraduatedOptimizationRegressor

pl.seed_everything(0)
#
module = GraduatedOptimizationRegressor.load_from_checkpoint('../../lightning_logs/version_15/checkpoints/epoch=4-step=4999.ckpt')
module.eval()
ds = Noisy2DPointsDataset(
    low=np.array([-1.0] * 2),
    high=np.array([1.0] * 2),
    num_points_range=(100, 101),
    batch_size=1,
    outliers_ratio_range=(0.8, 0.8),
    cov_eigenvalues_range=(0.02, 0.05),
    ds_size=50
)

losses_learned = []
loss_rule_based = []

for i in tqdm.tqdm(range(len(ds))):
    X, y = ds[i]
    X = torch.from_numpy(X)

    with torch.no_grad():
        theta = module(X)[-1][0].numpy()
    losses_learned.append(np.linalg.norm(theta - y))

    theta = graduated_optimization(X[0], kernel=WelschKernel()).numpy()
    loss_rule_based.append(np.linalg.norm(theta - y))
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
