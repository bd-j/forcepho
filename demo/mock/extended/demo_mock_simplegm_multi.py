# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs, otionally using HMC
# ---------

import sys, os
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho.sources import Star, SimpleGalaxy, Scene
from forcepho.likelihood import WorkPlan, make_image

from demo_utils import make_stamp, negative_lnlike_multi, Posterior


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



if __name__ == "__main__":

    nband = 1
    nsource = 2

    # --- Setup Scene and Stamp(s) ---

    # flux, ra, dec, q, pa(deg)
    if nband == 1:
        # Let's make two SimpleGalaxy in one band
        filters = ["band1"]
        sourcepars = [([10.], 11.0, 11.0, 0.7, 45),
                      ([15.], 15.0, 15.0, 0.7, 45)]
        upper = np.array(nsource * [20., 20., 20., 1.0,  np.pi/2.])
        lower = np.array(nsource * [2.,   2.,  5., 0.0, -np.pi/2])
    elif nband == 2:
        # This is what two bands would look like:
        filters = ["band1", "band2"]
        sourcepars = [([10., 12.], 5., 5., 0.7, 45),
                    ([15., 30.], 10., 10., 0.7, 45)]
        upper = np.array(nsource * [20., 40., 20., 20., 1.0,  np.pi/2.])
        lower = np.array(nsource * [2.,  5.,  2.,  5.,  0.0, -np.pi/2])


    # And two stamps
    stamp_kwargs = [{'size': (30., 30.), 'fwhm': 2.0, 'filtername': "band1"}]
    if nband == 1:
        # Add another stamp in the same band, but offset
        stamp_kwargs += [{'size': (30., 30.), 'fwhm': 2.0,
                          'filtername': "band1", 'offset': (-1.5, -1.5)}]
    if nband == 2:
        # Add another stamp in a different band
        stamp_kwargs += [{'size': (30., 30.), 'fwhm': 2.0,'filtername': "band2"}]
    
    scene, stamps, ptrue, label = setup_scene(galaxy=True, sourceparams=sourcepars,
                                              perturb=0.0, add_noise=True, snr_max=10.,
                                              filters=filters, stamp_kwargs=stamp_kwargs)

    nband = len(filters)
    nstamp = len(stamps)
    nsource = len(sourcepars)
    ndim = len(ptrue)

    # --- Set up posterior prob fns ----
    plans = [WorkPlan(stamp) for stamp in stamps]
    
    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans)
    nll_nograd = argfix(negative_lnlike_multi, scene=scene, plans=plans, grad=False)

    model = Posterior(scene, plans, upper=upper, lower=lower)

    # --- Optimization -----
    if False:
        p0 = ptrue.copy()
        from scipy.optimize import minimize
        def callback(x):
            print(x, nll(x))

        result = minimize(nll, p0 * 1.2, jac=True, bounds=None, callback=callback, method='BFGS',
                          options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
                                   'disp':True, 'iprint': 1, 'maxcor': 20})
        #result = minimize(nll_nograd, p0 * 1.2, jac=False, bounds=None, callback=callback,
        #                  options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
        #                           'disp':True, 'iprint': 1, 'maxcor': 20})
        
    
    # --- HMC Sampling -----
    if True:
        p0 = ptrue.copy()

        import hmc
        # initialize sampler and sample
        sampler = hmc.BasicHMC(model, verbose=False)
        eps = sampler.find_reasonable_epsilon(p0)
        iterations = 5000
        length = 50
        sigma_length = 10
        pos, prob, eps = sampler.sample(p0*1.2, iterations=iterations,
                                        epsilon=eps/5., length=length, sigma_length=sigma_length,
                                        store_trajectories=True)

        # --- Store sampler ---
        sampler.sourcepars = sourcepars
        sampler.stamp_kwargs = stamp_kwargs
        sampler.filters = filters
        sampler.offsets = [(0., 0.)]
        sampler.plans = plans
        sampler.scene = scene
        sampler.model = None
        import pickle
        with open("result_verysimple.pkl", "wb") as f:
            pickle.dump(sampler, f)
        sampler.model = model

        vals = pos

    # --- Plot resulting images ---
    if True:
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

    # --- Plot resulting chains/posteriors ---
    if True:
        ndim = nsource * 5
        tfig, taxes = pl.subplots(ndim, 2)
        for i in range(ndim):
            taxes[i, 0].plot(sampler.chain[:, i].flatten())
            taxes[i, 0].axhline(p0[i], linestyle=':', color='k')
            taxes[i, 1].hist(sampler.chain[:, i].flatten(), alpha=0.5, bins=30)
            taxes[i, 1].axvline(p0[i], linestyle=':', color='k')

        pnames = 'flux', 'RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)'
        [ax.set_xlabel(p) for ax, p in zip(taxes[:, 1], pnames)]


        tfig.tight_layout()
        import corner
        choose = [0, 5]
        cfig = corner.corner(sampler.chain[:, choose], labels=["$Flux_1$", "$Flux_2$"])
        cfig.savefig("demo_simple_fluxcovar.pdf")

        pl.show()
