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

    
    # --- Setup Scene and Stamp(s) ---

    # Let's make two SimpleGalaxies
    # flux, ra, dec, q, pa(deg)
    sourcepars = [([10., 12.], 5., 5., 0.7, 45),
                  ([15., 30.], 10., 10., 0.7, 45)]
    # And stamp(s)
    stamp_kwargs = [{'size': (30., 30.), 'fwhm': 2.0,
                     'filtername': "f1"},
                    {'size': (30., 30.), 'fwhm': 2.0,
                     'filtername': "f2", 'offset': (-4.5, -4.5)}
                    ]
    scene, stamps, ptrue, label = setup_scene(galaxy=True, sourceparams=sourcepars,
                                              filters=["f1", "f2"],
                                              perturb=0.0, add_noise=True,
                                              snr_max=10.,
                                              stamp_kwargs=stamp_kwargs)

    #sys.exit()
    #true_images = [make_image(scene, stamp, Theta=ptrue)[0] for stamp in stamps]
    
    # Set up (negative) likelihoods
    plans = [WorkPlan(stamp) for stamp in stamps]
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

        vals = result.x
        rfig, raxes = pl.subplots(len(stamps), 3, sharex=True, sharey=True)
        for i, stamp in enumerate(stamps):
            im, grad = make_image(scene, stamp, Theta=vals)
            raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
            raxes[i, 1].imshow(im.T, origin='lower')
            resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
            rfig.colorbar(resid, ax=raxes[i,:].tolist())
        
        labels = ['Data', 'Model', 'Data-Model']
        [ax.set_title(labels[i]) for i, ax in enumerate(raxes[0,:])]
        

    pl.show()