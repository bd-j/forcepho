# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho import gaussmodel as gm
from forcepho import psf
from forcepho.likelihood import model_image, set_galaxy_params


def negative_lnlike_stamp(theta, source=None, stamp=None, free_inds=slice(0, 3)):
    stamp.residual = stamp.pixel_values.flatten()
    thetas = [np.append(np.array(theta), np.array([1., 0., 0., 0.]))]
    #thetas[0] *= 100
    residual, partials = model_image(thetas, [source], stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials[free_inds, :], axis=-1)


def negative_lnlike_nograd(theta, source=None, stamp=None, free_inds=slice(0, 3)):
    stamp.residual = stamp.pixel_values.flatten()
    thetas = [np.append(np.array(theta), np.array([1., 0., 0., 0.]))]
    #thetas[0] *= 100
    residual, partials = model_image(thetas, [source], stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2)


def get_image(theta, source=None, stamp=None, free_inds=slice(0, 3)):
    stamp.residual = stamp.pixel_values.flatten() * 0.0
    thetas = [np.append(np.array(theta), np.array([1., 0., 0., 0.]))]
    #thetas[0] *= 100
    residual, partials = model_image(thetas, [source], stamp)
    return -residual, partials


if __name__ == "__main__":

    galaxy, new = False, True
    
    # get a stamp and put an image in it
    stamp = gm.PostageStamp()
    stamp.nx = 30
    stamp.ny = 30
    stamp.npix = int(stamp.nx * stamp.ny)
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])
    stamp.residual = np.zeros(stamp.npix)
    stamp.psf = psf.PointSpreadFunction()

    # get a source
    if galaxy:
        source = gm.Galaxy()
        source.ngauss = 4
        source.radii = np.arange(source.ngauss) * 0.5
        source.q = 0.5
        source.pa = np.deg2rad(30.)
        use = slice(0,5)
        theta = [1, 10., 10., 0.5, np.deg2rad(30.), 0., 0.]
        label = ['$\psi$', '$x$', '$y$', '$q$', '$\\varphi$']
        bounds = [(0, 1e4), (0., 30), (0, 30), (0, 1), (0, np.pi/2)]
    else:
        source = gm.Star()
        use = slice(0, 3)
        theta = [100, 10., 10., 1., 0., 0., 0.]
        label = ['$\psi$', '$x$', '$y$']
        bounds = [(-1e6, 1e6), (-1e5, 1e5), (-1e5, 1e5)]

    # Set source parameters
    set_galaxy_params(source, theta)
    # center the image on the source
    stamp.crval = np.array([source.ra, source.dec])
    stamp.crval += 0.5 # move off-center by a half pix

    # ---- Generate mock -------
    # ----- New code ------
    if new:
        ptrue = np.array(theta) * 1.25
        residual, partials = model_image([ptrue], [source], stamp,
                                        use_gradients=use)
        im = -residual
    
    
    #  --- OLD code ----
    if not new:
        gig = gm.convert_to_gaussians(source, stamp)
        gig = gm.get_gaussian_gradients(source, stamp, gig)

        for g in gig.gaussians.flat:
            im, partial_phi = gm.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat)
            # This matrix multiply can be optimized (many zeros)
            partials = np.matmul(g.derivs, partial_phi)



    stamp.pixel_values = -residual.reshape(stamp.nx, stamp.ny).copy()
    err = stamp.pixel_values.max() * 1e-2
    #err = np.sqrt(stamp.pixel_values.flatten())
    stamp.ierr = np.ones(stamp.npix) / err
    nll = argfix(negative_lnlike_stamp, source=source, stamp=stamp)
    nll_nograd = argfix(negative_lnlike_nograd, source=source, stamp=stamp)


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
        c = ax.imshow(im.reshape(stamp.nx, stamp.ny).T, origin='lower')
        ax.text(0.1, 0.85, 'Mock (I)'.format(label[i]), transform=ax.transAxes)
        fig.colorbar(c, ax=ax)

    # --- Optimization -----


    p0 = np.array(theta[use])
    from scipy.optimize import minimize
    def callback(x):
        print(x, nll(x))

    result = minimize(nll, p0 * 1.5, jac=True, bounds=None, callback=callback, method='BFGS',
                      options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                      )
    result_nograd = minimize(nll_nograd, p0 * 1.5, jac=False, bounds=None, callback=callback,
                      options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                      )

