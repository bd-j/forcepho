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

from demo_utils import Posterior, make_stamp, negative_lnlike_multi, numerical_image_gradients


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

    # --- Get Stamps ----
    print(len(stamp_kwargs))
    stamps = [make_stamp(**sk) for sk in stamp_kwargs]
    if splinedata is not None:
        for stamp in stamps:
            stamp.scale = np.array([[32.0, 0.0], [0.0, 32.0]])

    # --- Generate mock and add to stamp ---
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
    catname = "catalog_for_demo.dat"
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
        scales = nband * [0.5] + 2 * [0.2] + [0.05] + [0.05]
        scales = np.array(len(sourcepars) * scales)
        mass_matrix = 1.0 / scales**2
        #mass_matrix = None

        # Initialize sampler
        import hmc
        sampler = hmc.BasicHMC(model, verbose=False)
        sampler.ndim = len(p0)
        sampler.set_mass_matrix(mass_matrix)
        sampler.sourcepars = sourcepars
        sampler.stamp_kwargs = stamp_kwargs
        sampler.filters = filters
        sampler.offsets = offsets
        sampler.plans = plans
        sampler.scene = scene
        sampler.truths = ptrue.copy()

        
        eps = sampler.find_reasonable_stepsize(p0, epsilon_guess=1e-2)
        #eps = 0.01
        #sys.exit()
        length = 50
        sigma_length = 10
        # Burn-in
        pos, prob, grad = sampler.sample(p0*1.05, iterations=20, mass_matrix=mass_matrix,
                                        epsilon=eps / 2., length=length, sigma_length=sigma_length,
                                        store_trajectories=True)

        post_burnin = pos.copy()
        eps_run = sampler.find_reasonable_stepsize(pos, epsilon_guess=eps)
        #sys.exit()
        pos, prob, grad = sampler.sample(pos, iterations=100, mass_matrix=mass_matrix,
                                        epsilon=eps_run / 2., length=length, sigma_length=sigma_length,
                                        store_trajectories=True)


        best = sampler.chain[sampler.lnp.argmax()]
        from phoplot import plot_chain
        out = plot_chain(sampler, show_trajectories=True, equal_axes=True, source=1)
        #import corner
        #cfig = corner.corner(sampler.chain[10:], truths=ptrue.copy(), labels=label, show_titles=True)
        #pl.show()

        sampler.model = None
        import pickle
        with open("mock_simplegm_{}stamp_snr{:02.0f}.pkl".format(len(stamps), snr_max), "wb") as f:
            pickle.dump(sampler, f)
        sampler.model = model

        
    # --- Plot results ---
    if False:
        # plot the data and model
        from phoplot import plot_model_images
        rfig, raxes = plot_model_images(best, scene, stamps[:6])
        pl.show()

    if False:
        # plot movies
        from phoplot import hmc_movie
        sourceid = 1
        iterations = [1, 2, 3, 4, 5, 10, 15, 20]
        psub = np.array([0, 1, 2, 3, 4, 5])
        pinds = scene.sources[0].nparam * sourceid + psub
        stamps_to_show = [stamps[0], stamps[3]]
        label = np.array(scene.sources[sourceid].parameter_names)[psub]
        hmc_movie(sampler, iterations, stamps_to_show, pinds, label)#, x=slice(10, 40), y=slice(10, 40))
        

    pl.show()
