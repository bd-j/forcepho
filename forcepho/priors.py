# -*- coding: utf-8 -*-

import numpy as np

__all__ = ["LogPrior",
           "ExpBeta", "Barrier"]


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


class ExpBeta(LogPrior):
    """Used for optimization; heavily weights the model against values near min and max
    """
    def __init__(self, min, max, alpha=-0.9):
        self.min = min
        self.max = max
        self.range = self.max - self.min
        self.alpha = alpha
        assert self.alpha > -1, f"Alpha ({self.alpha}) must be greater than -1"

    def _lnprior(self, x):
        """This is ln-probability as a beta distribution
        """
        return  -(x**self.alpha * (1 - x) ** self.alpha)

    def _lnprior_grad(self, x):
        return self.alpha * (2*x - 1) * (-(x - 1) * x)**(self.alpha - 1)


class Barrier(LogPrior):
    """Used for optimization; heavily weights the model against values near min and max

    n.b. does not seem to actually work
    """
    def __init__(self, min, max, alpha=-2, edgewidth=0.05):
        self.min = min
        self.max = max
        self.range = self.max - self.min
        self.alpha = alpha
        self.edgewidth = edgewidth
        raise NotImplementedError

    def _lnprior(self, x):
        lpr = np.zeros_like(x)
        z = (x - 1) / self.edgewidth + 1
        y = 1 - x/ self.edgewidth
        g = z > 0
        lpr[g] = lpr[g] + (1 - 1/z[g])**self.alpha
        g = (y > 0) #& (y < 1)
        #print(1 - 1/y[g])
        #print(x)
        #print(y)
        lpr[g] = lpr[g] + (1 - 1/y[g])**self.alpha
        return  lpr

    def _lnprior_grad(self, x):
        lpr_grad = np.zeros_like(x)
        z = (x - 1) / self.edgewidth + 1
        y = 1 - x/ self.edgewidth
        g = z > 0
        lpr_grad[g] = lpr_grad[g] + self.alpha * (1 - 1/z[g])**(self.alpha-1) * (1/z[g]**2)
        g = y > 0
        lpr_grad[g] = lpr_grad[g] - self.alpha * (1 - 1/y[g])**(self.alpha-1) * (1/y[g]**2)
        bad = (z > 1) | (y > 1)
        lpr_grad[bad] = np.inf
        return lpr_grad / self.edgewidth

class Beta(LogPrior):
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


class TruncatedCauchy(LogPrior):

    def sample(self, loc=0, scale=1, a=-1, b=1, size=None):
        """
        Generate random samples from a truncated Cauchy distribution.

        `loc` and `scale` are the location and scale parameters of the distribution.
        `a` and `b` define the interval [a, b] to which the distribution is to be
        limited.

        With the default values of the parameters, the samples are generated
        from the standard Cauchy distribution limited to the interval [-1, 1].
        """
        ua = np.arctan((a - loc)/scale)/np.pi + 0.5
        ub = np.arctan((b - loc)/scale)/np.pi + 0.5
        U = np.random.uniform(ua, ub, size=size)
        rvs =  loc + scale * np.tan(np.pi*(U - 0.5))
        return rvs


def test_grad(pl, ptype=Barrier):

    lnprior = ptype(10, 30)
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
    ax.plot(qq, -lnp_g - nll_g_num)

def test_min_with_prior(pl, ptype=Barrier):

    from forcepho.model import BoundedTransform, Transform

    ndim = 3
    mu = np.ones(ndim) * 0.5
    lo = np.ones(ndim)*(1)
    hi = np.ones(ndim)*3.
    lnprior = ptype(lo, hi)
    transform = BoundedTransform(lo, hi)
    #transform = Transform(ndim)

    def nll(z):
        q = transform.transform(z)
        #print(q)
        ll = -0.5 * np.sum((q - mu)**2)
        ll_grad = -(q-mu)  #dlike/dq
        #lpr, lpr_grad = lnprior(q)
        lpr, lpr_grad = 0.0, 0.0
        lnp = ll + lpr #+ transform.lndetjac(z)
        jac = transform.jacobian(z)
        lnp_grad = (ll_grad + lpr_grad) * jac #+ transform.lndetjac_grad(z)
        return -lnp, -lnp_grad

    def model(z):
        nlp, nlp_grad = nll(z)
        lnp = -nlp + transform.lndetjac(z)
        lnp_grad = -nlp_grad + transform.lndetjac_grad(z)
        return lnp, lnp_grad

    z0 = transform.inverse_transform(np.ones(ndim) * 2.5)

    # Optimization
    from scipy.optimize import minimize
    result = minimize(nll, z0, jac=True, method="BFGS")
    print(result.message)
    print(result.nfev)
    print(transform.transform(result.x))

    # HMC
    from littlemcmc.sampling import _sample as sample_one
    from littlemcmc import NUTS
    from littlemcmc import QuadPotentialFull, QuadPotentialFullAdapt, QuadPotentialDiagAdapt


    step = NUTS(logp_dlogp_func=model,
                model_ndim=ndim)
    trace, stats = sample_one(logp_dlogp_func=model,
                              model_ndim=ndim, start=z0, step=step,
                              draws=2024, tune=4048, chain=0,
                              progressbar=True, random_seed=0xDEADBEEF)
    chain = transform.transform(trace.T)