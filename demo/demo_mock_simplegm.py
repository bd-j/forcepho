# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho import gaussmodel as gm
from demo_utils import Scene, make_stamp, negative_lnlike_stamp, negative_lnlike_nograd, make_image


def numerical_image_gradients(theta0, delta):

    dI_dp = 0
    for i, (p, dp) in enumerate(zip(theta0, delta)):
        theta = theta0.copy()
        imlo, _ = make_image(theta, scene, stamp)
        theta[i] += dp
        imhi, _ = make_image(theta, scene, stamp)
        dI_dp.append((imhi - imlo) / (dp))

    return np.array(dI_dp)


def setup_scene(galaxy=False, fudge=1.0, fwhm=1.0, offset=0.0, size=(), add_noise=False):


    stamp = make_stamp(size, fwhm, offset=offset)
    
    # --- Get a Source and Scene -----
    if galaxy:
        source = gm.Galaxy()
        source.ngauss = 4
        source.radii = np.arange(source.ngauss) * 0.5
        source.q = 0.5
        source.pa = np.deg2rad(30.)
        theta = [100., 10., 10., 0.5, np.deg2rad(30.)]
        label = ['$\psi$', '$x$', '$y$', '$q$', '$\\varphi$']
        bounds = [(0, 1e4), (0., 30), (0, 30), (0, 1), (0, np.pi/2)]
    else:
        source = gm.Star()
        theta = [100., 10., 10.]
        label = ['$\psi$', '$x$', '$y$']
        bounds = [(-1e6, 1e6), (-1e5, 1e5), (-1e5, 1e5)]

    scene = Scene()
    scene.sources = [source]
    scene.set_params(theta)


    # --- Generate mock  and add to stamp ---
    ptrue = np.array(theta) * fudge
    true_image, partials = make_image(ptrue, scene, stamp)
    stamp.pixel_values = true_image.copy()
    err = stamp.pixel_values.max() * 1e-2
    #err = np.sqrt(stamp.pixel_values.flatten())
    stamp.ierr = np.ones(stamp.npix) / err

    if add_noise:
        noise = np.random.normal(0, err, size=(stamp.nx, stamp.ny))
        stamp.pixel_values += noise

    
    return scene, stamp, ptrue, label


if __name__ == "__main__":

    # Get a scene and a stamp at some parameters
    scene, stamp, ptrue, label = setup_scene(galaxy=True, fwhm=2.0, fudge=1.25)
    true_image, partials = make_image(ptrue, scene, stamp)
    
    # Set up likelihoods
    nll = argfix(negative_lnlike_stamp, scene=scene, stamp=stamp)
    nll_nograd = argfix(negative_lnlike_nograd, scene=scene, stamp=stamp)

    #sys.exit()

    # --- Chi2 on a grid ------
    # needs to be debugged
    if False:
        mux = np.linspace(47, 53., 100)
        muy = np.linspace(47, 53., 100)
        flux = np.linspace(3000, 5000., 10)
        chi2 = np.zeros([len(mux), len(muy), len(flux)])

        for i, x in enumerate(mux):
            for j, y in enumerate(muy):
                for k, f in enumerate(flux):
                    theta = np.array([f, x, y])
                    chi2[i,j,k] = nll(theta)[0]

        sys.exit()


    # ---- Plot mock image and gradients thereof -----
    if True:
        fig, axes = pl.subplots(3, 2)
        for i, ddtheta in enumerate(partials):
            ax = axes.flat[i+1]
            c = ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny).T, origin='lower')
            ax.text(0.1, 0.85, '$\partial I/\partial${}'.format(label[i]), transform=ax.transAxes)
            fig.colorbar(c, ax=ax)

        ax = axes.flat[0]
        c = ax.imshow(true_image.T, origin='lower')
        ax.text(0.1, 0.85, 'Mock (I)'.format(label[i]), transform=ax.transAxes)
        fig.colorbar(c, ax=ax)


    # ---- Test Gradients ------
    if True:
        delta = np.ones_like(ptrue) * 1e-6
        #numerical
    
    
    # --- Optimization -----
    if True:
        p0 = ptrue.copy()
        from scipy.optimize import minimize
        def callback(x):
            print(x, nll(x))

        result = minimize(nll, p0 * 1.2, jac=True, bounds=None, callback=callback, method='BFGS',
                        options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                        )
        result_nograd = minimize(nll_nograd, p0 * 1.2, jac=False, bounds=None, callback=callback,
                        options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                        )

