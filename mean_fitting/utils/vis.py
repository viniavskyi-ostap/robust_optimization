import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_loss_on_scales(loss_history, num_iter_per_scale):
    plt.plot(np.arange(loss_history.shape[0]), loss_history)
    plt.xticks(np.arange(len(loss_history)))
    plt.show()

if __name__ == '__main__':
    plot_loss_on_scales(None, [None] * 6)