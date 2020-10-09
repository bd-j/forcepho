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
except ImportError:
    Op = object
    from argparse import Namespace
    tt = Namespace(fscalar=np.float32, fvector=np.ndarray,
                   dscalar=np.float64, dvector=np.ndarray,
                   scalar=np.float, vector=np.ndarray)

try:
    import pycuda
    import pycuda.autoinit
except ImportError:
    pass

from .kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS
from .likelihood import lnlike_multi, make_image, WorkPlan


__all__ = ["Posterior", "GPUPosterior", "CPUPosterior",
           "LogLikeWithGrad",
           "Transform", "BoundedTransform"]


class Posterior:
    """Base class for models.  Includes the basic caching mechanism for
    probabilities and gradients, as well as a numerical check on the gradients.
    """

    def __init__(self, **kwargs):
        '''
        Initializes fields that are common to all subclasses.
        
        Parameters
        ----------
        transform: Transform, optional
            The transform to use. Must specify this, or `lower` and `upper`.
        lower: array-like, optional
            Lower bounds for a BoundedTransform. `upper` must also be specified
            if given. Mutually exclusive with `transform`.
        upper: array-like, optional
            Upper bounds for a BoundedTransform. `lower` must also be specified
            if given. Mutually exclusive with `transform`.
        '''
        self.ncall = 0
        
        if (np.any(kwargs.get('lower')) and np.any(kwargs.get('upper'))) == bool(kwargs.get('transform')):
            raise ValueError('Must specify either "transform" or ("lower","upper")')
        
        transform = kwargs.pop('transform',None)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = BoundedTransform(kwargs.pop('lower'), kwargs.pop('upper'))
            
        if kwargs:
            raise ValueError(f'Passed argument(s) {list(kwargs.keys())} to Posterior that were not understood')
            

    def evaluate(self, z):
        """Method to actually evaluate ln-probability and its gradient, at the
        sampling aparameter position `z`, and cache these quantities. Other
        values may be cached or actions performed at this time.  This should be
        subclassed.

        Populates the following attributes
        * _z - the parameter posiiton
        * _lnp - the ln-likelihood at z
        * - lnp_grad - the gradient of the ln-likelihood with respect to z
        * ncall - the number of likelihood calls is incremented by 1

        Parameters
        ----------
        z : ndarray of shape (ndim,)
            A parameter vector.
        """
        self._z = z
        self._lnp = 0
        self._lnp_grad = 0
        self.ncall += 1
        raise(NotImplementedError)

    def lnprob(self, z):
        """Get the ln-probability at parameter position `z`.  This uses the
        cached value of the ln-probability if z has not chenged, and otherwise
        calls the `evaluate` method.

        Parameters
        ----------
        z : ndarray of shape (ndim,)
            The parameter position (in the sampling parameter space)

        Returns
        -------
        lnp : np.float
            The ln-probability at the parameter position `z`
        """
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        """Get the gradients of the ln-probability with respect to `z`, at
        parameter position `z`.  This uses the cached value of the
        ln-probability if z has not chenged, and otherwise calls the `evaluate`
        method.

        Parameters
        ----------
        z : ndarray of shape (ndim,)
            The parameter position (in the unconstrained sampling parameter
            space)

        Returns
        -------
        lnp_grad : ndarray of shape (ndim,)
            The gradient of the ln-probability with repsect to `z`, at the
            parameter position `z`
        """
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad

    def lnprior(self, q):
        """The prior probability and its gradient with respect to the scene
        parameters. By default this returns 0.0 for both, but this method can be
        overridden.

        Parameters
        ----------
        q : ndarray of shape (ndim,)
            The array of parameters in the scene space

        Returns
        -------
        lpr : float
            The ln of the prior probability for the parameters `q`

        lpr_grad : ndarray of shape (ndim,) or 0.0
            The gradient of `lpr` with respect to the scene parameters `q`
        """
        return 0.0, 0.0

    def make_transform(self, z):
        """Transform the sampling parameters z in the unconstrained space to
        the constrained scene parameters q using the `transform` attribute. This
        also caches the jacobian of the transform, the determinant of that
        jacobian, and the gradient of the determinant.

        Parameters
        ----------
        z : ndarray of shape (ndim,)
            The parameters in the (unconstrained) sampling space

        Returns
        -------
        q : ndarray
            The parameters in the scene space
        """
        if self.transform is not None:
            self._jacobian = self.transform.jacobian(z)
            self._lndetjac = self.transform.lndetjac(z)
            self._lndetjac_grad = self.transform.lndetjac_grad(z)
            return self.transform.transform(z)
        else:
            self._jacobian = 1.
            self._lndetjac = 0
            self._lndetjac_grad = 0
            return np.array(z)

    def nll(self, z):
        """A shortcut for the negative ln-likelihood, for use with
        minimization algorithms. Returns both the NLL and it's gradient
        """
        if np.any(z != self._z):
            self.evaluate(z)
        return -self._lnp, -self._lnp_grad

    def lnprob_and_grad(self, z):
        """Some samplers want a single function to return lnp, dlnp.
        This is that.
        """
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp, self._lnp_grad

    def residuals(self, z):
        raise(NotImplementedError)

    def check_grad(self, z, delta=1e-5):
        """Compare numerical gradients to analytic gradients.

        Parameters
        ----------
        z : ndarray, shape (ndim,)
            The parameter location at which to check gradients

        delta : float or ndarray, optional, default=1e-5
            Change in parameter values to use for calculating
            numerical gradients

        Returns
        -------
        dlnp : ndarray, shape (ndim,)
            the analytic gradients

        dlnp_num : ndarray, shape (ndim,)
            The numerical gradients
        """
        z0 = z.copy()
        dlnp = self.lnprob_grad(z)

        delta = np.zeros_like(dlnp) + delta
        dlnp_num = np.zeros(len(z0), dtype=np.float64)
        for i, dp in enumerate(delta):
            theta = z0.copy()
            theta[i] -= dp
            imlo = self.lnprob(theta)
            theta[i] += 2 * dp
            imhi = self.lnprob(theta)
            dlnp_num[i] = ((imhi - imlo) / (2 * dp))
        return dlnp, dlnp_num


class GPUPosterior(Posterior):

    """A Posterior subclass that uses a GPU to evaluate the likelihood
    and its gradients.
    """

    def __init__(self, proposer, scene=None, lnprior=None, transform=None,
                 name="", print_interval=1000, verbose=False, debug=False,
                 logging=False, **kwargs):
        
        super().__init__(transform=transform, **kwargs)
        
        # --- Assign ingredients ---
        self.proposer = proposer
        self.scene = scene if scene else proposer.patch.scene
        if lnprior is not None:
            self.lnprior = lnprior

        # --- initialize some things ---
        self.ncall = 0
        self._z = -99
        self._q = -99
        self.mhalf = np.array(-0.5, dtype=np.float64)

        # --- logging/debugging ---
        self.debug = debug
        self.logging = logging
        self.verbose = verbose
        self.print_interval = print_interval
        self.name = name
        if self.logging:
            self.pos_history = []
            self.lnp_history = []
            self.grad_history = []

    def evaluate(self, z):
        """Compute the log-likelihood and its gradient, using the GPU propsoer
        object.  These are then cached.

        Parameters
        ----------
        z : ndarray of shape (ndim,)
            The parameter vector in the unconstrained space.  This will be
            transformed to the constrained space and fed to the scene object to
            set scene parameters and generate a GPU proposal.
        """

        # --- Transform to scene parameters and get proposal vector & prior ---
        q = self.make_transform(z)
        self.scene.set_all_source_params(q)
        proposal = self.scene.get_proposal()
        lpr, lpr_grad = self.lnprior(q)
        if self.debug:
            print(proposal["fluxes"])

        # --- send to gpu and collect result ---
        ret = self.proposer.evaluate_proposal(proposal)
        if len(ret) == 3:
            chi2, chi2_derivs, self._residuals = ret
        else:
            chi2, chi2_derivs = ret

        # --- Turn into log-like and accumulate grads correctly ---
        # note type conversion before sum to avoid loss of significance
        ll = self.mhalf * np.array(chi2.astype(np.float64).sum(), dtype=np.float64)
        ll_grad = self.mhalf * self.stack_grad(chi2_derivs)

        # --- Caching and computation of final lnp and lnp_grad ---
        self.ncall += 1

        # these are not actually required
        if self.logging:
            self._lnlike = ll
            self._lnlike_grad = ll_grad
            self._lnprior = lpr
            self._lnprior_grad = lpr_grad

        self._lnp = ll + lpr + self._lndetjac
        self._lnp_grad = (ll_grad + lpr_grad) * self._jacobian + self._lndetjac_grad
        self._q = q
        self._z = z

        # --- Logging/Debugging ---
        if self.debug:
            print("chi2: {}".format(chi2))

        if self.logging:
            self.pos_history.append(z)
            self.lnp_history.append(ll)
            self.grad_history.append(ll_grad)

        if self.verbose:
            if np.mod(self.ncall, self.print_interval) == 0.:
                print("-------\n {} @ {}".format(self.name, self.ncall))
                print("q:", q)
                print("lnlike:", ll)
                print("lnlike_grad:", ll_grad)

    def stack_grad(self, chi2_derivs):
        """The chi2_derivs is returned with shape NBAND, NACTIVE, NPARAMS.

        Final output should be [flux11, flux12, ..., flux1Nb, ra1, dec1, ..., rh1,
                                flux21, flux22, ..., flux2Nb, ra2, dec2, ..., rh2]
        """
        nsources = self.proposer.patch.n_sources
        nbands = self.proposer.patch.n_bands
        grads = np.zeros([nsources, nbands + (NPARAMS-1)], dtype=np.float64)
        for band, derivs in enumerate(chi2_derivs):
            # shape params
            grads[:, nbands:] += derivs[:, 1:]
            # flux params
            grads[:, band] += derivs[:, 0]
            if self.debug:
                print(band, derivs[0, :])

        return grads.reshape(-1)

    def residuals(self, z):
        """Return residual images
        """
        assert self.proposer.patch.return_residual
        self.scene.set_all_source_params(z)
        proposal = self.scene.get_proposal()
        ret = self.proposer.evaluate_proposal(proposal)
        _, _, self._residuals = ret
        return self._residuals


class CPUPosterior(Posterior):

    def __init__(self, scene=None, plans=[], stamps=[], lnlike=lnlike_multi,
                 lnlike_kwargs={"source_meta": True}, lnprior=None, transform=None):

        # --- Assign ingredients ---
        self.scene = scene
        self.transform = transform
        self.lnlike = lnlike
        self.lnlike_kwargs = lnlike_kwargs
        if len(plans) == 0:
            self.plans = [WorkPlan(stamp) for stamp in stamps]
        else:
            self.plans = plans
        if lnprior is not None:
            self.lnprior = lnprior

        # --- initialize some things ---
        self.ncall = 0
        self._z = -99
        self._q = -99

    def evaluate(self, z):
        """Evaluate and chache the ln-probability for the given parameter
        vector, and its gradient.

        Parameters
        ----------
        z : ndarray of shape (self.ndim,)
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime, i.e. the
        transformation is the identity.
        """
        q = self.make_transform(z)
        ll, ll_grad = self.lnlike(q, scene=self.scene, plans=self.plans)
        lpr, lpr_grad = self.lnprior(q)

        self.ncall += 1
        self._lnlike = ll
        self._lnlike_grad = ll_grad
        self._lnprior = lpr
        self._lnprior_grad = lpr_grad

        self._lnp = ll + lpr + self._lndetjac
        # TODO: In the general case this should be a dot product of the
        # gradient vector with a Jacobian matrix, but for now the jacobian is diagonal so this works
        self._lnp_grad = (ll_grad + lpr_grad) * self._jacobian + self._lndetjac_grad
        self._q = q
        self._z = z

    def residuals(self, Theta=None):
        """Construct model image (residuals) for the given parameter vecotrs.
        """
        if Theta:
            self.scene.set_all_source_params(Theta)
        self._residuals = [plan.stamp.pixel_values - make_image(self.scene, plan.stamp)[0]
                           for plan in self.plans]
        return self._residuals


class ConstrainedTransformedPosterior(CPUPosterior):


    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.  This is only useful for bd-j/hmc backends.

        Parameters
        ----------
        theta : The parameter vector

        Returns
        -------
        theta : The new theta vector

        sign : A vector of multiplicative signs for the momenta

        flag : A flag for if the values are still out of bounds.
        """
        upper = np.zeros_like(theta) + self.upper
        lower = np.zeros_like(theta) + self.lower

        # initially no flips
        sign = np.ones_like(theta)
        # pretend we started out-of-bounds to force at least one check
        oob = True
        while oob:
            above = theta > upper
            theta[above] = 2 * upper[above] - theta[above]
            sign[above] *= -1
            below = theta < lower
            theta[below] = 2 * lower[below] - theta[below]
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
