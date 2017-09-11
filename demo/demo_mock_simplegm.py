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


class Scene(object):

    def set_params(self, theta):
        # Add all the unused (fixed) galaxy parameters
        t = np.array(theta).copy()
        if len(t) == 3:
            # Star
            self.params = [np.append(t, np.array([1., 0., 0., 0.]))]
            self.free_inds = slice(0, 3)
        elif len(t) == 5:
            # Galaxy
            self.params = [np.append(np.array(t), np.array([0., 0.]))]
            self.free_inds = slice(0, 5)
        else:
            print("theta vector {} not a valid length: {}".format(theta, len(theta)))




def negative_lnlike_stamp(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials[scene.free_inds, :], axis=-1)


def negative_lnlike_nograd(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2)


def make_image(theta, scene=None, stamp=None):
    stamp.residual = np.zeros(stamp.npix)
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    return -residual.reshape(stamp.nx, stamp.ny), partials[scene.free_inds, :]


if __name__ == "__main__":

    galaxy = False
    
    # get a stamp and put an image in it
    stamp = gm.PostageStamp()
    stamp.nx = 30
    stamp.ny = 30
    stamp.npix = int(stamp.nx * stamp.ny)
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])
    stamp.residual = np.zeros(stamp.npix)
    stamp.psf = psf.PointSpreadFunction()

    # get a source and scene
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

    # center the image on the source
    stamp.crval = np.array([theta[1], theta[2]])
    #stamp.crval += 0.5 # move off-center by a half pix

    # ---- Generate mock -------
    ptrue = np.array(theta) * 1.25
    true_image, partials = make_image(ptrue, scene, stamp)
    stamp.pixel_values = true_image.copy()
    err = stamp.pixel_values.max() * 1e-2
    #err = np.sqrt(stamp.pixel_values.flatten())
    stamp.ierr = np.ones(stamp.npix) / err

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

    # --- Optimization -----


    p0 = np.array(theta)
    from scipy.optimize import minimize
    def callback(x):
        print(x, nll(x))

    result = minimize(nll, p0 * 1.5, jac=True, bounds=None, callback=callback, method='BFGS',
                      options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                      )
    result_nograd = minimize(nll_nograd, p0 * 1.5, jac=False, bounds=None, callback=callback,
                      options={'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                      )

