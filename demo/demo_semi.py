# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys, os, time
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import SimpleGalaxy, Galaxy, Scene
from forcepho.likelihood import WorkPlan, make_image

from demo_utils import make_stamp, negative_lnlike_multi, numerical_image_gradients


psfnames = {"F090W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F115W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F150W": os.path.join(paths.psfmixture, 'f150w_ng6_em_random.p'),
            "F200W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            "F277W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            }


def setup_scene(sourceparams=[(1.0, 5., 5., 0.7, 30., 1.0, 0.05)],
                splinedata=None, perturb=0,
                filters=['dummy'],
                add_noise=False, snr_max=100.,
                stamp_kwargs=[]):

    # --- Get Sources and a Scene -----
    sources = []
    for pars in sourceparams:
        flux, x, y, q, pa, n, rh = np.copy(pars)
        if splinedata is not None:
            s = Galaxy(filters=filters, splinedata=splinedata)
            s.sersic = n
            s.rh = rh
        else:
            s = SimpleGalaxy(filters=filters)
        s.flux = flux
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()
    label = []

    # get stamps
    print(len(stamp_kwargs))
    stamps = [make_stamp(**sk) for sk in stamp_kwargs]
    if splinedata is not None:
        for stamp in stamps:
            stamp.scale = np.array([[32.0, 0.0], [0.0, 32.0]])

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
        print(Theta)
        t = time.time()
        nll, nll_grad = negative_lnlike_multi(Theta, scene=self.scene, plans=self.plans)
        print(time.time() - t)
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


def read_catalog(catname):
    with open(catname, "r") as f:
        lines = f.readlines()
    nband = int(lines[0].replace('#', '').split()[1])
    filters = lines[1].replace('#', '').split()[:nband]
    filters = [f.upper() for f in filters]
    i = 2
    dat = []
    while lines[i][0] != '#':
        dat.append(np.array([float(c) for c in lines[i].split()]))
        i += 1
    #dat = np.array(dat)
    i += 1
    offsets = []
    while (lines[i][0] != '#'):
        offsets.append([float(c) for c in lines[i].split()])
        i += 1
    offsets = np.array(offsets)

    # flux, ra, dec, q, pa(deg), n, sersic
    params = [(d[:nband], d[nband], d[nband+1], d[nband+2], d[nband+3], 0.0, 0.0)
              for d in dat if len(d) >= (nband + 4)]

    return params, filters, offsets   


if __name__ == "__main__":

    
    # --- Setup Scene and Stamp(s) ---
    catname = "catalog_for_demo_full.dat"
    sourcepars, filters, offsets = read_catalog(catname)
    #sys.exit()
    
    stamp_kwargs = []
    for f in filters:
        for o in offsets:
            p = psfnames[f]
            sk = {'size': (30, 30), 'psfname': p, 'filtername': f, 'offset': o}
            stamp_kwargs.append(sk)

    # flux, ra, dec, q, pa(rad), n, sersic
    nband = len(filters)
    up = nband * [15] + [20., 20., 1.0,  np.pi/2.]
    lo = nband * [1] + [5.,  5., 0.0, -np.pi/2.]
    upper = np.array(len(sourcepars) * up)
    lower = np.array(len(sourcepars) * lo)


    scene, stamps, ptrue, label = setup_scene(sourceparams=sourcepars,
                                              splinedata=None,  # paths.galmixture,
                                              perturb=0.0, add_noise=True,
                                              snr_max=10., filters=filters,
                                              stamp_kwargs=stamp_kwargs)


    # --- Set up posterior prob fns ----
    plans = [WorkPlan(stamp) for stamp in stamps]
    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans)

    #upper = np.zeros(5) + 1000
    #lower = np.zeros(5) - 1000
    model = Posterior(scene, plans, upper=upper, lower=lower)

    # --- show model ----
    if True:
        rfig, raxes = pl.subplots(len(filters), len(offsets), sharex=True, sharey=True)
        raxes = np.atleast_2d(raxes)
        for i, stamp in enumerate(stamps):
            raxes.flat[i].imshow(stamp.pixel_values.T, origin='lower')
        pl.show()

    #sys.exit()
    # --- Gradient Check ---
    if False:
        delta = np.ones_like(ptrue) * 1e-7
        #numerical
        grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
        image, grad = make_image(scene, stamps[0], Theta=ptrue)
        fig, axes = pl.subplots(len(ptrue), 3, sharex=True, sharey=True)
        for i in range(len(ptrue)):
            g = grad[i,:].reshape(stamp.nx, stamp.ny)
            c = axes[i, 0].imshow(grad_num[i,:,:].T, origin='lower')
            fig.colorbar(c, ax=axes[i, 0])
            c = axes[i, 1].imshow(g.T, origin='lower')
            fig.colorbar(c, ax=axes[i, 1])
            c = axes[i, 2].imshow((grad_num[i,:,:] - g).T, origin='lower')
            fig.colorbar(c, ax=axes[i, 2])

        axes[0, 0].set_title('Numerical')
        axes[0, 1].set_title('Analytic')
        axes[0, 2].set_title('N - A')
    
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
        p0 = ptrue.copy() #model.renormalize_theta(ptrue)

        import hmc
            #initialize sampler and sample
        sampler = hmc.BasicHMC(verbose=False)
        eps = sampler.find_reasonable_epsilon(p0, model, epsilon_guess=1e-2)
        #eps = 0.01
        #sys.exit()
        iterations = 2000
        length = 50
        sigma_length = 10
        pos, prob, eps = sampler.sample(p0*1.05, model, iterations=iterations,
                                        epsilon=eps/5., length=length, sigma_length=sigma_length,
                                        store_trajectories=True)

        vals = pos  # model.renormalize_theta(pos)
        #rfig, raxes = pl.subplots(len(stamps), 3, sharex=True, sharey=True)
        #raxes = np.atleast_2d(raxes)
        #for i, stamp in enumerate(stamps):
        #    im, grad = make_image(scene, stamp, Theta=vals)
        #    raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
        #    raxes[i, 1].imshow(im.T, origin='lower')
        #    resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
        #    rfig.colorbar(resid, ax=raxes[i,:].tolist())
        
        #labels = ['Data', 'Model', 'Data-Model']
        #[ax.set_title(labels[i]) for i, ax in enumerate(raxes[0,:])]


        ndim = scene.sources[0].nparam
        tfig, taxes = pl.subplots(ndim, 2)
        #traj_end = np.arange(0, iterations*length, length)
        for i in range(ndim):
            l, n = [], 0
            for traj in sampler.trajectories:
                taxes[i, 0].plot(np.arange(len(traj)) + n,
                                 traj[:, i], color='r')
                n += len(traj)
                l.append(n)
            taxes[i, 0].plot(np.array(l), sampler.chain[:, i],'o')

            taxes[i, 0].axhline(p0[i], linestyle=':', color='k')
            taxes[i, 1].hist(sampler.chain[:, i], alpha=0.5, bins=30)
            taxes[i, 1].axvline(p0[i], linestyle=':', color='k')

        pnames = filters + ['RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)', 'n', 'r$_h$']
        [ax.set_xlabel(p) for ax, p in zip(taxes[:, 1], pnames)]

        sampler.sourcepars = sourcepars
        sampler.stamp_kwargs = stamp_kwargs
        sampler.filters = filters
        sampler.offsets = offsets
        sampler.plans = plans
        sampler.scene = scene
        sampler.truths = ptrue.copy()

        import pickle
        with open("semi_results_snr10.pkl", "wb") as f:
            pickle.dump(sampler, f)

        #tfig.tight_layout()
        #import corner
        #cfig = corner.corner(
        

    pl.show()
