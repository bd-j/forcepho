# ----------
# Script to fit a single point source in a single Guitarra simulated image.
#-----------

import sys, os
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import Star, Scene
from forcepho.likelihood import WorkPlan, make_image

from demo_utils import negative_lnlike_multi, make_real_stamp


if __name__ == "__main__":

    inpixels = False  # whether to fit in pixel coordinates or celestial coordinates
    imname = os.path.join(paths.starsims, 'sim_cube_F090W_487_001.slp.fits')
    psfname = os.path.join(paths.psfmixture, 'f090_ng6_em_random.p')

    # --- Build the postage stamp ----
    # ra_init, dec_init = 53.116342, -27.80352 # has a hole
    # add_stars     53.115299   -27.803508  1407.933314  1194.203114  18.000       4562.19      48983.13       49426
    #ra_init, dec_init = 53.115325, -27.803518
    # keep in mind 1pixel ~ 1e-5 degrees
    ra_init, dec_init = 53.115295, -27.803501
    stamp = make_real_stamp(imname, (ra_init, dec_init), center_type='celestial',
                            size=(100, 100), psfname=psfname)
    #stamp.ierr = stamp.ierr.flatten() / 10

    if inpixels:
        # override the WCS so coordinates are in pixels
        # The scale matrix D
        stamp.scale = np.eye(2)
        stamp.dpix_dsky = np.eye(2)
        # The sky coordinates of the reference pixel
        stamp.crval = np.zeros([2])
        # The pixel coordinates of the reference pixel
        stamp.crpix = np.zeros([2])

    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    stamp.psf.covariances = np.matmul(T, np.matmul(stamp.psf.covariances, T.T))
    stamp.psf.means = np.matmul(stamp.psf.means, T)

    # --- get the Scene ---
    sources = [Star(filters=["F090W"])]
    label = ['flux', 'x', 'y']
    scene = Scene(sources)
    plans = [WorkPlan(stamp)]

    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans)
    nll_nograd = argfix(negative_lnlike_multi, scene=scene, plans=plans, grad=False)

    # --- Initialize ---
    theta_init = np.array([stamp.pixel_values.sum() * 1.0, ra_init, dec_init])
    if inpixels:
        world = np.array([ra_init, dec_init, 0])
        hdr = fits.getheader(imname)
        ast = apy_wcs.WCS(hdr)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2] - stamp.lo
        theta_init = np.array([stamp.pixel_values.sum() * 0.5, center[0], center[1]])
    image_init, partials = make_image(scene, stamp, Theta=theta_init)

    # --- Plot initial value ---
    if True:
        fig, axes = pl.subplots(3, 2, sharex=True, sharey=True)
        ax = axes.flat[0]
        i = ax.imshow(stamp.pixel_values.T, origin='lower')
        ax.text(0.1, 0.9, 'Sim. Data', transform=ax.transAxes)
        ax = axes.flat[1]
        i = ax.imshow(image_init.T, origin='lower')
        ax.text(0.1, 0.9, 'Initial Model', transform=ax.transAxes)
        for i, ddtheta in enumerate(partials):
            ax = axes.flat[i+2]
            ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny).T, origin='lower')
            ax.text(0.1, 0.9, '$\partial I/\partial {}$'.format(label[i]), transform=ax.transAxes)
        pl.show()


    # --- Chi2 on a grid ---
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


    # --- Optimization ---
    if True:
        def callback(x):
            #nf += 1
            print(x, nll(x))

        p0 = theta_init.copy()
        p0[0] = 4500. #34.44
        if inpixels:
            p0[1] = 50. #48.1
            p0[2] = 50. #51.5
        bounds = [ (0, 1e4), (0., 100), (0, 100)]
        from scipy.optimize import minimize
        result = minimize(nll, p0, jac=True, bounds=None, callback=callback,
                        options={'ftol': 1e-20, 'gtol': 1e-12, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20})

        result_nograd = minimize(nll_nograd, p0, jac=False, bounds=None, callback=callback,
                                 options={'ftol': 1e-20, 'gtol': 1e-12, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                                 )

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
