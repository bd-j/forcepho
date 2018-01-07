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

from demo_utils import make_stamp, numerical_image_gradients, negative_lnlike_multi


def setup_scene(galaxy=False, sourceparams=[(1.0, 5., 5., 0.7, 30.)],
                filters=['dummy'],
                add_noise=False, snr_max=100.,
                stamp_kwargs={}):


    # get a stamp
    stamp = make_stamp(**stamp_kwargs)
    
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
    ptrue = np.array(theta)
    true_image, partials = make_image(scene, stamp, Theta=ptrue)
    stamp.pixel_values = true_image.copy()
    err = stamp.pixel_values.max() / snr_max
    #err = np.sqrt(err**2 + stamp.pixel_values.flatten())
    err = np.ones(stamp.npix) * err
    stamp.ierr = np.ones(stamp.npix) / err

    if add_noise:
        noise = np.random.normal(0, err)
        stamp.pixel_values += noise.reshape(stamp.nx, stamp.ny)

    return scene, stamp, ptrue, label


if __name__ == "__main__":

    # Get a scene and a stamp at some parameters
    # Let's make two SimpleGalaxies
    # flux, ra, dec, q, pa(deg)
    sourcepars = [([10.], 5., 5., 0.7, 45),
                  ([15.], 10., 10., 0.7, 45)]
    # And one stamp
    stamp_kwargs = {'size': (30., 30.), 'fwhm':2.0, 'snr_max': 100}
    scene, stamp, ptrue, label = setup_scene(galaxy=True, sourceparams=sourcepars,
                                             add_noise=True,
                                             stamp_kwargs=stamp_kwargs)

    sys.exit()
    true_image, partials = make_image(scene, stamp, Theta=ptrue)
    
    # Set up (negative) likelihoods
    plans = [WorkPlan(stamp)]
    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans)
    nll_nograd = argfix(negative_lnlike_multi, scene=scene, plans=plans, grad=False)


    # --- Optimization -----
    if True:
        p0 = ptrue.copy()
        from scipy.optimize import minimize
        def callback(x):
            print(x, nll(x))

        result = minimize(nll, p0 * 1.2, jac=True, bounds=None, callback=callback, method='BFGS',
                        options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
                                 'disp':True, 'iprint': 1, 'maxcor': 20}
                        )
        #result_nograd = minimize(nll_nograd, p0 * 1.2, jac=False, bounds=None, callback=callback,
        #                options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
        #                         'disp':True, 'iprint': 1, 'maxcor': 20}
        #                )

        mim, partials = make_image(scene, stamp, Theta=result.x)
        dim = stamp.pixel_values
        
        fig, axes = pl.subplots(1, 3, sharex=True, sharey=True, figsize=(13.75, 4.25))
        images = [dim, mim, dim-mim]
        labels = ['Data', 'Model', 'Data-Model']
        for k, ax in enumerate(axes):
            c = ax.imshow(images[k].T, origin='lower')
            pl.colorbar(c, ax=ax)
            ax.set_title(labels[k])
        

    pl.show()
