import numpy as np


class BasicKernel:
    def _call_unscaled(self, x):
        raise NotImplementedError

    def _get_weight_unscaled(self, x):
        raise NotImplementedError

    def __call__(self, x, scale=1.):
        return scale ** 2 * self._call_unscaled(x / scale)

    def get_weight(self, x, scale=1.):
        return self._get_weight_unscaled(x / scale)


class WelschKernel(BasicKernel):
    def _call_unscaled(self, x):
        """
        Calculate Welsch kernel function on input residuals
        Args:
            x: array of residuals

        Returns:
            robust loss over residuals
        """
        return 0.5 * (1 - (-x ** 2).exp())

    def _get_weight_unscaled(self, x):
        """
        Weight function of Welsch kernel evaluated at x
        Args:
            x: array of residuals

        Returns:
            weights for IRLS estimation
        """
        return (-x ** 2).exp()


class WelschKernelNumpy(BasicKernel):
    def _call_unscaled(self, x):
        """
        Calculate Welsch kernel function on input residuals
        Args:
            x: array of residuals

        Returns:
            robust loss over residuals
        """
        return 0.5 * (1 - np.exp(-x ** 2))

    def _get_weight_unscaled(self, x):
        """
        Weight function of Welsch kernel evaluated at x
        Args:
            x: array of residuals

        Returns:
            weights for IRLS estimation
        """
        return np.exp(-x ** 2)


class PowerKernel(BasicKernel):
    def __init__(self, p):
        super(PowerKernel, self).__init__()
        self.p = p

    def _call_unscaled(self, x):
        """
        Calculate Power kernel function on input residuals
        Args:
            x: array of residuals

        Returns:
            loss over residuals
        """
        return x ** self.p

    def _get_weight_unscaled(self, x):
        """
        Weight function of Power kernel evaluated at x
        Args:
            x: array of residuals

        Returns:
            weights for IRLS estimation
        """
        return self.p * (x ** (self.p - 2))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    kernel = WelschKernel()
    x = np.linspace(-3, 3, 300)
    plt.plot(x, kernel(x))
    kernel.scale = 1.1
    plt.plot(x, kernel(x))

    plt.show()
