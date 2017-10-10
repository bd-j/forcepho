# ----------
# Script to fit a single point source in a single Guitarra simulated image.
#-----------

import sys, os
from copy import deepcopy
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as fits
from astropy import wcs as apy_wcs

from forcepho import paths
from forcepho import psf as pointspread
from forcepho.gaussmodel import Star
from forcepho.data import PostageStamp
from demo_utils import Scene, negative_lnlike_stamp, negative_lnlike_nograd, make_image


def make_stamp(imname, center=(None, None), size=(None, None),
               center_type='pixels', psfname=None, fwhm=1.0):
    """Make a postage stamp around the given position using the given image name
    """
    data = fits.getdata(imname)
    hdr = fits.getheader(imname)
    crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
    crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                   [hdr['CD2_1'], hdr['CD2_2']]])

    # Pull slices and transpose to get to an axis order that makes sense to me
    # and corresponds with the wcs keyword ordering
    im = data[0, :, :].T
    err = data[1, :, :].T

    # ---- Extract subarray -----
    center = np.array(center)
    # here we get the center coordinates in pixels (accounting for the transpose above)
    if center_type == 'celestial':
        world = np.append(center, 0)
        ast = apy_wcs.WCS(hdr)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2]
    # here is much mystery ---
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    crpix_stamp = np.floor(0.5 * size)
    crval_stamp = crpix_stamp + lo
    W = np.eye(2)
    if center_type == 'celestial':
        crval_stamp = ast.wcs_pix2world(np.append(crval_stamp, 0.)[None,:], 0)[0, :2]
        W[0, 0] = np.cos(np.deg2rad(crval_stamp[-1]))

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds]
    stamp.ierr = 1./err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0
    stamp.ierr = stamp.ierr.flatten()

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval_stamp
    stamp.scale = np.matmul(np.linalg.inv(CD), W)
    stamp.pixcenter_in_full = center
    stamp.lo = lo

    # --- Add the PSF ---
    if psfname is not None:
        import pickle
        with open(psfname, 'rb') as pf:
            pdat = pickle.load(pf)

        oversample, center = 8, 504 - 400
        answer = pdat[6][2]
        stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=center)
    else:
        stamp.psf = pointspread.PointSpreadFunction()
        stamp.psf.covaraniaces *= fwhm/2.355
    
    # --- Add extra information ---
    stamp.full_header = dict(hdr)    
    return stamp


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
    stamp = make_stamp(imname, (ra_init, dec_init), center_type='celestial',
                       size=(100, 100), psfname=psfname)
    #stamp.ierr = stamp.ierr.flatten() / 10

    if inpixels:
        # override the WCS so coordinates are in pixels
        # The scale matrix D
        stamp.scale = np.eye(2)
        # The sky coordinates of the reference pixel
        stamp.crval = np.zeros([2])
        # The pixel coordinates of the reference pixel
        stamp.crpix = np.zeros([2])

    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    stamp.psf.covariances = np.matmul(T, np.matmul(stamp.psf.covariances, T.T))
    stamp.psf.means = np.matmul(stamp.psf.means, T)

    # --- get the Scene ---
    scene = Scene()
    sources = [Star()]
    scene.sources = sources
    label = ['flux', 'x', 'y']

    nll = argfix(negative_lnlike_stamp, scene=scene, stamp=stamp)
    nll_nograd = argfix(negative_lnlike_nograd, scene=scene, stamp=stamp)

    # --- Initialize ---
    theta_init = np.array([stamp.pixel_values.sum() * 1.0, ra_init, dec_init])
    if inpixels:
        world = np.array([ra_init, dec_init, 0])
        hdr = fits.getheader(imname)
        ast = apy_wcs.WCS(hdr)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2] - stamp.lo
        theta_init = np.array([stamp.pixel_values.sum() * 0.5, center[0], center[1]])
    image_init, partials = make_image(theta_init, scene, stamp)

    # --- Plot initial value ---
    if True:
        fig, axes = pl.subplots(3, 2, sharex=True, sharey=True)
        ax = axes.flat[0]
        i = ax.imshow(stamp.pixel_values.T, origin='lower')
        ax.text(0.1, 0.9, 'Sim. Data', transform=ax.transAxes)
        ax = axes.flat[1]
        i = ax.imshow(image_init.T, origin='lower')
        ax.text(0.1, 0.9, 'Initial Model', transform=ax.transAxes)
        for i, ddtheta in enumerate(partials[scene.free_inds, :]):
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
        bounds = [(0, 1e4), (0., 100), (0, 100)]
        from scipy.optimize import minimize
        result = minimize(nll, p0, jac=True, bounds=None, callback=callback,
                        options={'ftol': 1e-20, 'gtol': 1e-12, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20})

        result_nograd = minimize(nll_nograd, p0, jac=False, bounds=None, callback=callback,
                                 options={'ftol': 1e-20, 'gtol': 1e-12, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20}
                                 )

        mim, partials = make_image(result.x, scene, stamp)
        dim = stamp.pixel_values

        fig, axes = pl.subplots(1, 3, sharex=True, sharey=True, figsize=(13.75, 4.25))
        images = [dim, mim, dim-mim]
        labels = ['Data', 'Model', 'Data-Model']
        for k, ax in enumerate(axes):
            c = ax.imshow(images[k].T, origin='lower')
            pl.colorbar(c, ax=ax)
            ax.set_title(labels[k])

        pl.show()
