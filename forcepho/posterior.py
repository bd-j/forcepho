import numpy as np
from .likelihood import lnlike_multi

__all__ = ["Posterior"]


class Posterior(object):

    def __init__(self, scene, plans, upper=np.inf, lower=-np.inf, verbose=False):
        self.scene = scene
        self.plans = plans
        self._theta = -99
        self.lower = lower
        self.upper = upper
        self.verbose = verbose
        self.ncall = 0

    def evaluate(self, theta):
        Theta = self.complete_theta(theta)
        if self.verbose:
            print(Theta)
            t = time.time()
        ll, ll_grad = lnlike_multi(Theta, scene=self.scene, plans=self.plans)
        lpr, lpr_grad = self.ln_prior_prob(Theta)
        if self.verbose:
            print(time.time() - t)
        self.ncall += 1
        self._lnlike = ll
        self._lnlike_grad = ll_grad
        self._lnprior = lpr
        self._lnprior_grad = lpr_grad
        self._lnp = ll + lpr
        self._lnp_grad = ll_grad + lpr_grad
        self._theta = Theta

    def ln_prior_prob(self, theta):
        return 0.0, np.zeros(len(theta))
        
    def lnprob(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp

    def lnprob_grad(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp_grad

    def complete_theta(self, theta):
        return theta

    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.

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
