# -*- coding: utf-8 -*-

import numpy as np

class LogPrior:

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max
        self.range = self.max - self.min

    def __call__(self, q):
        """Compute and return the ln-prior-probability at q, and the gradient of
        ln-prior_probability with respect to q.
        """
        x = self._scale(q)
        lnp = self._lnprior(x)
        lnp_grad = self._lnprior_grad(x) / self.range

        return np.sum(lnp), lnp_grad

    def _scale(self, q):
        return (q - self.min) / self.range

    def _lnprior(self, x):
        return 0

    def _lnprior_grad(self, x):
        return np.zeros_like(x)


class Skin(LogPrior):
    """Used for optimization; heavily weights the model against values near min and max
    """
    def __init__(self, min, max, alpha=-0.9):
        self.min = min
        self.max = max
        self.range = self.max - self.min
        self.alpha = alpha
        assert self.alpha > -1, f"Alpha ({self.alpha}) must be greater than -1"

    def _lnprior(self, x):
        return  -(x**self.alpha * (1 - x) ** self.alpha)

    def _lnprior_grad(self, x):
        return self.alpha * (2*x - 1) * (-(x - 1) * x)**(self.alpha - 1)


class Beta(LogPrior):
    pass


class Spline(LogPrior):
    pass


class Normal(LogPrior):
    pass


class GenNormal(LogPrior):
    pass


class HalfNormal(LogPrior):
    pass


class StudentT(LogPrior):
    pass


class LogUniform(LogPrior):
    pass



def test_grad(pl):

    lnprior = Skin(10, 30)
    qq = np.linspace(10, 30, 1000)
    lnp = np.array([lnprior(x)[0] for x in qq])
    lnp_g = np.array([lnprior(x)[1] for x in qq])
    nll_g_num = np.gradient(-lnp)/np.gradient(qq)

    pl.ion()
    fig, ax = pl.subplots()
    ax.plot(qq, -lnp)
    ax.plot(qq, -lnp_g)
    #nll_g_num = np.diff(-lnp)/np.diff(qq)
    ax.plot(qq, nll_g_num)
    fig, ax = pl.subplots()
    ax.plot(qq, -lnp_g / nll_g_num)