#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""model.py

These are the Posterior objects that take a proposal and evaluate its
probabiility, storing the ln-probablity and gradients thereof.

There is also a class that wraps such model objects in theano ops, and classes
for transforming parameters while keeping track of the Jacobians of the
transform.
"""

import numpy as np
try:
    import theano
    theano.gof.compilelock.set_lock_status(False)
    import theano.tensor as tt
    Op = tt.Op
except(ImportError):
    Op = object
    from argparse import Namespace
    tt = Namespace(fscalar=np.float32, fvector=np.ndarray,
                   dscalar=np.float64, dvector=np.ndarray,
                   scalar=np.float, vector=np.ndarray)

try:
    import pycuda
    import pycuda.autoinit
except(ImportError):
    pass

from .kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS
from .likelihood import lnlike_multi, make_image, WorkPlan


__all__ = ["Posterior", "GPUPosterior", "CPUPosterior",
           "LogLikeWithGrad",
           "Transform", "BoundedTransform"]


class Posterior:
    """Abstract class for models.  Includes the basic caching mechanism for
    probabilities and gradients.
    """

    def __init__(self):
        raise(NotImplementedError)

    def evaluate(self, z):
        self._z = z
        self._lnp = 0
        self._lnp_grad = 0
        raise(NotImplementedError)

    def lnprob(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad

    def nll(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return -self._lnp, -self._lnp_grad

    def residuals(self, z):
        raise(NotImplementedError)


class GPUPosterior(Posterior):

    def __init__(self, proposer, scene, name="", verbose=False, debug=False):
        self.proposer = proposer
        self.scene = scene
        self.ncall = 0
        self._z = -99
        self.verbose = verbose
        self.debug = debug
        self.name = name

    def evaluate(self, z):
        """
        :param z:
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime; the transformation
        is the identity.
        """
        self.scene.set_all_source_params(z)
        proposal = self.scene.get_proposal()
        if self.debug:
            print(proposal["fluxes"])

        # send to gpu and collect result
        ret = self.proposer.evaluate_proposal(proposal)
        if len(ret) == 3:
            chi2, chi2_derivs, self._residuals = ret
        else:
            chi2, chi2_derivs = ret

        mhalf = np.array(-0.5, dtype=np.float64)
        # turn into log-like and accumulate grads correctly
        ll = mhalf * np.array(chi2.sum(), dtype=np.float64)
        ll_grad = mhalf * self.stack_grad(chi2_derivs)
        if self.debug:
            print("chi2: {}".format(chi2))

        if self.verbose:
            if np.mod(self.ncall, 1000) == 0.:
                print("-------\n {} @ {}".format(self.name, self.ncall))
                print(z)
                print(ll)
                print(ll_grad)

        self.ncall += 1
        self._lnp = ll
        self._lnp_grad = ll_grad
        self._z = z

    def stack_grad(self, chi2_derivs):
        """The chi2_derivs is returned with shape NBAND, NACTIVE, NPARAMS.

        Final output should be [flux11, flux12, ..., flux1Nb, ra1, dec1, ..., rh1,
                                flux21, flux22, ..., flux2Nb, ra2, dec2, ..., rh2]
        """
        nsources = self.proposer.patch.n_sources
        nbands = self.proposer.patch.n_bands
        grads = np.zeros([nsources, nbands + (NPARAMS-1)])
        for band, derivs in enumerate(chi2_derivs):
            # shape params
            grads[:, nbands:] += derivs[:, 1:]
            # flux params
            grads[:, band] += derivs[:, 0]
            if self.debug:
                print(band, derivs[0, :])

        return grads.reshape(-1)

    def residuals(self, z):
        assert self.proposer.patch.return_residuals
        self.scene.set_all_source_params(z)
        proposal = self.scene.get_proposal()
        ret = self.proposer.evaluate_proposal(proposal)
        _, _, self._residuals = ret
        return self._residuals


class CPUPosterior(Posterior):

    def __init__(self, scene=None, plans=[], stamps=[], lnlike=lnlike_multi,
                 lnlike_kwargs={"source_meta": True}):
        self.lnlike = lnlike
        self.lnlike_kwargs = lnlike_kwargs
        if len(plans) == 0:
            self.plans = [WorkPlan(stamp) for stamp in stamps]
        else:
            self.plans = plans
        self.scene = scene
        self.ncall = 0
        self._z = -99

    def evaluate(self, z):
        """
        Evaluate the probability for the given parameter vector

        :param z:
            The sampling parameters which have a prior
            distribution attached.

        """
        Theta = z
        ll, ll_grad = self.lnlike(Theta, scene=self.scene, plans=self.plans,
                                  **self.lnlike_kwargs)
        self.ncall += 1
        self._lnp = ll
        self._lnp_grad = ll_grad
        self._z = z

    def residuals(self, Theta):
        """Construct model images for the given parameter vecotrs.
        """
        self.scene.set_all_source_params(Theta)
        self._residuals = np.array([make_image(self.scene, plan)[0]
                                    for plan in self.plans])
        return self._residuals


class ConstrainedTransformedPosterior(CPUPosterior):

    def __init__(self, scene=None, plans=[], stamps=[], lnlike=lnlike_multi,
                 lnprior=None, transform=None, upper=np.inf, lower=-np.inf,
                 verbose=False):
        self.scene = scene
        if len(plans) == 0:
            self.plans = [WorkPlan(stamp) for stamp in stamps]
        else:
            self.plans = plans
        self.lnlike = lnlike
        if lnprior is not None:
            self.lnprior = lnprior
        self.T = transform
        self.ncall = 0
        self._theta = -99
        self._z = -99
        self.lower = lower
        self.upper = upper

    def evaluate(self, z):
        """
        :param z:
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime, i.e. the
        transformation is the identity.
        """
        Theta = self.transform(z)
        ll, ll_grad = self.lnlike(Theta, scene=self.scene, plans=self.plans)
        lpr, lpr_grad = self.lnprior(Theta)

        self.ncall += 1
        self._lnlike = ll
        self._lnlike_grad = ll_grad
        self._lnprior = lpr
        self._lnprior_grad = lpr_grad

        self._lnp = ll + lpr + self._lndetjac
        self._lnp_grad = (ll_grad + lpr_grad) * self._jacobian + self._lndetjac_grad
        self._theta = Theta
        self._z = z

    def lnprior(self, Theta):
        return 0.0, 0.0

    def transform(self, z):
        if self.T is not None:
            self._jacobian = self.T.jacobian(z)
            self._lndetjac = self.T.lndetjac(z)
            self._lndetjac_grad = self.T.lndetjac_grad(z)
            return self.T.transform(z)
        else:
            self._jacobian = 1.
            self._lndetjac = 0
            self._lndetjac_grad = 0
            return np.array(z)

    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.  This is only useful for bd-j/hmc backends.

        :param theta:
            The parameter vector

        :returns theta:
            the new theta vector

        :returns sign:
            a vector of multiplicative signs for the momenta

        :returns flag:
            A flag for if the values are still out of bounds.
        """
        # initially no flips
        sign = np.ones_like(theta)
        # pretend we started out-of-bounds to force at least one check
        oob = True
        while oob:
            above = theta > self.upper
            theta[above] = 2 * self.upper[above] - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2 * self.lower[below] - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
        return theta, sign, oob


class ModelGradOp(Op):
    """Wraps the Posterior object lnprob_grad() method in a theano tensor
    operation
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        z, = inputs
        ll_grads = self.model.lnprob_grad(z)
        outputs[0][0] = ll_grads


class LogLikeWithGrad(Op):
    """Wraps the Posterior object lnprob() and lnprob_grad() methods in theano
    tensor operations
    """

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, model):
        self.model = model
        self.GradObj = ModelGradOp(self.model)

    def perform(self, node, inputs, outputs):
        z, = inputs
        logl = self.model.lnprob(z)
        outputs[0][0] = np.array(logl)

    def grad(self, inputs, g):
        z, = inputs
        return [g[0] * self.GradObj(z)]

# --- TRANSFORMS ---

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def logit(y):
    return -np.log(1./y - 1.)

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Transform(object):

    def __init__(self, ndim):
        self.ndim = ndim

    def transform(self, z):
        return np.array(z)

    def jacobian(self, z):
        return np.ones(len(z))

    def lndetjac(self, z):
        return np.log(np.abs(np.product(self.jacobian(z))))
        # return np.sum(np.log(self.jacobian(z)))

    def lndetjac_grad(self, z):
        return np.zeros(len(z))

    def inverse_transform(self, Theta):
        return np.array(Theta)


class BoundedTransform(Transform):

    def __init__(self, lower, upper):
        self.lower = np.atleast_1d(lower)
        self.upper = np.atleast_1d(upper)
        assert len(self.upper) == len(self.lower)
        self.ndim = len(self.upper)

    @property
    def range(self):
        return self.upper - self.lower

    def transform(self, z):
        """Transform from the sampling variable `z` to the constrained
        variable `Theta`.  I.e., this is the function g s.t. Theta = g(z)

        This uses the sigmoid function
        """
        y = sigmoid(z)
        Theta = self.lower + y * (self.range)
        return Theta

    def jacobian(self, z):
        return self.range * sigmoid(z) * (1.0 - sigmoid(z))

    def lndetjac_grad(self, z):
        return (1 - 2.0 * sigmoid(z))

    def inverse_transform(self, Theta):
        y = (Theta - self.lower) / self.range
        z = logit(y)
        return z
