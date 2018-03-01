# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys, os
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho.sources import Star, SimpleGalaxy, Scene
from forcepho.likelihood import WorkPlan, make_image

from demo_utils import make_stamp, negative_lnlike_multi


def setup_scene(galaxy=False, sourceparams=[(1.0, 5., 5., 0.7, 30.)],
                perturb=0,
                filters=['dummy'],
                add_noise=False, snr_max=100.,
                stamp_kwargs=[]):


    # get a stamp
    stamps = [make_stamp(**sk) for sk in stamp_kwargs]
    
    # --- Get Sources and a Scene -----
    if galaxy:
        ngauss = 1
        sources = []
        for (flux, x, y, q, pa) in sourceparams:
            s = SimpleGalaxy(filters=filters)
            s.flux = flux
            s.ra = x
            s.dec = y
            s.q = q
            s.pa = np.deg2rad(pa)
            s.radii = np.arange(ngauss) * 0.5 + 1.0
            sources.append(s)

    else:
        sources = []
        for (x, y, q, pa) in sourcelocs:
            s = Star(filters=filters)
            s.ra = x
            s.dec = y            
            sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()
    label = []

    # --- Generate mock  and add to stamp ---
    ptrue = theta * np.random.normal(1.0, perturb, size=theta.shape)
    for stamp in stamps:
        true_image, _ = make_image(scene, stamp, Theta=ptrue)
        stamp.pixel_values = true_image.copy()
        err = stamp.pixel_values.max() / snr_max
        #err = np.sqrt(err**2 + stamp.pixel_values.flatten())
        err = np.ones(stamp.npix) * err
        stamp.ierr = np.ones(stamp.npix) / err
        if add_noise:
            noise = np.random.normal(0, err)
            stamp.pixel_values += noise.reshape(stamp.nx, stamp.ny)

    return scene, stamps, ptrue, label

class Posterior(object):

    def __init__(self, scene, plans, upper=np.inf, lower=-np.inf):
        self.scene = scene
        self.plans = plans
        self.theta = -99
        self.lower = lower
        self.upper = upper

    def evaluate(self, Theta):
        nll, nll_grad = negative_lnlike_multi(Theta, scene=self.scene, plans=self.plans)
        self._lnp = -nll
        self._lnp_grad = -nll_grad
        self._theta = Theta

    def lnprob(self, Theta):
        if np.any(Theta != self.theta):
            self.evaluate(Theta)
        return self._lnp

    def lnprob_grad(self, Theta):
        if np.any(Theta != self.theta):
            self.evaluate(Theta)
        return self._lnp_grad

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


if __name__ == "__main__":

    
    # --- Setup Scene and Stamp(s) ---

    # Let's make one SimpleGalaxy in one and
    # flux, ra, dec, q, pa(deg)
    filters = ["flux"]
    #sourcepars = [([30.], 10., 10., 0.7, 45)]
    nsource = 2
    sourcepars = [([10.], 11.0, 11.0, 0.7, 45),
                  ([15], 15.0, 15.0, 0.7, 45)]

    upper = np.array(nsource * [20., 20., 20., 1.0, np.pi/2.])
    lower = np.array(nsource * [2., 2., 5., 0.0, -np.pi/2])
    
    
    
    # And one stamp
    stamp_kwargs = [{'size': (30., 30.), 'fwhm': 2.0,
                     'filtername': "flux"},
                    {'size': (30., 30.), 'fwhm': 2.0,
                     'filtername': "flux"}]
    scene, stamps, ptrue, label = setup_scene(galaxy=True, sourceparams=sourcepars,
                                              perturb=0.0, add_noise=True,
                                              snr_max=10., filters=filters,
                                              stamp_kwargs=stamp_kwargs)

    
    # --- Set up posterior prob fns ----
    plans = [WorkPlan(stamp) for stamp in stamps]
    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans)

    #upper = np.zeros(5) + 1000
    #lower = np.zeros(5) - 1000
    model = Posterior(scene, plans, upper=upper, lower=lower)


    # --- Optimization -----
    if False:
        p0 = ptrue.copy()
        from scipy.optimize import minimize
        def callback(x):
            print(x, nll(x))

        result = minimize(nll, p0 * 1.2, jac=True, bounds=None, callback=callback, method='BFGS',
                        options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
                                 'disp':True, 'iprint': 1, 'maxcor': 20}
                        )
        vals = result.x
        rfig, raxes = pl.subplots(len(stamps), 3, sharex=True, sharey=True)
        raxes = np.atleast_2d(raxes)
        for i, stamp in enumerate(stamps):
            im, grad = make_image(scene, stamp, Theta=vals)
            raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
            raxes[i, 1].imshow(im.T, origin='lower')
            resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
            rfig.colorbar(resid, ax=raxes[i,:].tolist())
        
        labels = ['Data', 'Model', 'Data-Model']
        [ax.set_title(labels[i]) for i, ax in enumerate(raxes[0,:])]
        

    
    # --- Sampling -----
    if True:
        p0 = ptrue.copy()

        import hmc
            #initialize sampler and sample
        sampler = hmc.BasicHMC(model, verbose=False)
        eps = sampler.find_reasonable_epsilon(p0)
        #sys.exit()
        iterations = 5000
        length = 50
        sigma_length = 10
        pos, prob, eps = sampler.sample(p0*1.2, iterations=iterations,
                                        epsilon=eps/5., length=length, sigma_length=sigma_length,
                                        store_trajectories=True)

        vals = pos
        rfig, raxes = pl.subplots(len(stamps), 3, sharex=True, sharey=True)
        raxes = np.atleast_2d(raxes)
        for i, stamp in enumerate(stamps):
            im, grad = make_image(scene, stamp, Theta=vals)
            raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
            raxes[i, 1].imshow(im.T, origin='lower')
            resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
            rfig.colorbar(resid, ax=raxes[i,:].tolist())
        
        labels = ['Data', 'Model', 'Data-Model']
        [ax.set_title(labels[i]) for i, ax in enumerate(raxes[0,:])]

        ndim = nsource * 5
        tfig, taxes = pl.subplots(ndim, 2)
        for i in range(ndim):
            taxes[i, 0].plot(sampler.chain[:, i].flatten())
            taxes[i, 0].axhline(p0[i], linestyle=':', color='k')
            taxes[i, 1].hist(sampler.chain[:, i].flatten(), alpha=0.5, bins=30)
            taxes[i, 1].axvline(p0[i], linestyle=':', color='k')

        pnames = 'flux', 'RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)'
        [ax.set_xlabel(p) for ax, p in zip(taxes[:, 1], pnames)]

        pl.show()

        #tfig.tight_layout()
        # import corner
        # choose = [0, 5]
        # cfig = corner.corner(sampler.chain[:, choose], labels=["$Flux_1$", "$Flux_2$"])
        # cfig.savefig("demo_simple_fluxcovar.pdf")

        sampler.sourcepars = sourcepars
        sampler.stamp_kwargs = stamp_kwargs
        sampler.filters = filters
        sampler.offsets = [(0., 0.)]
        sampler.plans = plans
        sampler.scene = scene

        import pickle
        with open("result_verysimple.pkl", "wb") as f:
            pickle.dump(sampler, f)

        

