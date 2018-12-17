import numpy as np
from .likelihood import lnlike_multi

__all__ = ["Posterior", "LogLikeWithGrad",
           "Transform", "BoundedTransform"]


class Posterior(object):

    def __init__(self, scene, plans, lnlike=lnlike_multi, lnprior=None,
                 transform=None, upper=np.inf, lower=-np.inf, verbose=False):
        self.scene = scene
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
        
    def lnprob(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad

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
        #initially no flips
        sign = np.ones_like(theta)
        oob = True #pretend we started out-of-bounds to force at least one check
        #print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2*self.upper[above] - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2*self.lower[below] - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            #print('theta_out ={0}'.format(theta))
        return theta, sign, oob


class ModelGradOp(tt.Op):
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


class LogLikeWithGrad(tt.Op):
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
        #return np.sum(np.log(self.jacobian(z)))
        
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
