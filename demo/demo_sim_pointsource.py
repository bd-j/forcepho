import sys
from copy import deepcopy
from functools import partial

import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as fits
from astropy import wcs

from forcepho.gaussmodel import PostageStamp, Star
from forcepho import psf as pointspread
from forcepho.likelihood import model_image


class Scene(object):

    def set_params(self, theta, stamps=None):
        # Add all the unused (fixed) galaxy parameters
        t = theta.copy()
        t[0] *= 100
        self.params = [np.append(t, np.array([1., 0., 0., 0.]))]
        self.free_inds = slice(0, 3)


def make_stamp(imname, center=(None, None), size=(None, None),
               center_type='pixels'):
    """Make a postage stamp around the given position using the given image name
    """
    data = fits.getdata(imname)
    hdr = fits.getheader(imname)
    crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
    crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    distortion = np.array([[hdr['CD1_1'], hdr['CD1_2']],
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
        #hdr.update(NAXIS=2)
        ast = wcs.WCS(hdr)
        center = ast.wcs_world2pix(center[0], center[1], 0, 0)[:2]
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    # only valid for simple tan plane projetcions (i.e. no distortions)
    crpix_stamp = crpix - lo

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds]
    stamp.ierr = 1./err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval
    stamp.distortion = distortion
    stamp.pixcenter_in_full = center

    # --- Add extra information ---
    stamp.full_header = dict(hdr)
    
    return stamp


def negative_lnlike_stamp(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials[scene.free_inds, :], axis=-1)


if __name__ == "__main__":

    
    imname = '/Users/bjohnson/Projects/nircam/mocks/image/star/sim_cube_F090W_487_001.slp.fits'
    psfname = '/Users/bjohnson/Codes/image/forcepho/forcepho/gauss_mix/f090_ng6_em_random.p'

    # --- Build the postage stamp ----
    # ra_init, dec_init = 53.116342, -27.80352 # has a hole
    ra_init, dec_init = 53.115325, -27.803518
    # add_stars     53.115299   -27.803508  1407.933314  1194.203114  18.000       4562.19      48983.13       49426
    stamp = make_stamp(imname, (ra_init, dec_init), (100, 100), center_type='celestial')
    stamp.ierr = stamp.ierr.flatten() / 1000
    
    # override the WCS so coordinates are in pixels
    # The distortion matrix D
    stamp.distortion = np.eye(2)
    # The sky coordinates of the reference pixel
    stamp.crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    stamp.crpix = np.zeros([2])

    # --- Add the PSF ---
    from psf import *
    import pickle
    with open(psfname, 'rb') as pf:
        pdat = pickle.load(pf)

    oversample, center = 8, 504 - 400
    answer = pdat[6][2]
    stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=center)
    

    # --- get the Scene ---
    scene = Scene()
    sources = [Star()]
    scene.sources = sources

    # --- Initialize and plot ----
    #theta_init = np.array([stamp.pixel_values.sum() * 0.5, stamp.nx/2, stamp.ny/2])
    theta_init = np.array([stamp.pixel_values.sum() * 1.0, 48.1, 51.5])
    scene.set_params(theta_init)
    stamp.residual = np.zeros(stamp.npix)
    resid, partials = model_image(scene.params, sources, stamp)

    label = ['flux', 'x', 'y']
    
    if True:
        fig, axes = pl.subplots(3, 2, sharex=True, sharey=True)
        ax = axes.flat[0]
        i = ax.imshow(stamp.pixel_values.T, origin='lower')
        ax.text(0.1, 0.9, 'Sim. Data', transform=ax.transAxes)
        ax = axes.flat[1]
        i = ax.imshow(-resid.reshape(stamp.nx, stamp.ny).T, origin='lower')
        ax.text(0.1, 0.9, 'Initial Model', transform=ax.transAxes)
        for i, ddtheta in enumerate(partials[scene.free_inds, :]):
            ax = axes.flat[i+2]
            ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny).T, origin='lower')
            ax.text(0.1, 0.9, '$\partial I/\partial {}$'.format(label[i]), transform=ax.transAxes)
        pl.show()


    nll = partial(negative_lnlike_stamp, scene=scene, stamp=stamp)
    #chisq_init = nll(theta_init)


    # --- Chi2 on a grid -----
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


    # ---- Optimization ------

    if False:
        def callback(x):
            #nf += 1
            print(x, nll(x))

        p0 = theta_init.copy()
        p0[0] = 45.6 #34.44
        p0[1] = 50. #48.1
        p0[2] = 50. #51.5
        bounds = [(0, 1e4), (0., 100), (0, 100)]
        from scipy.optimize import minimize
        result = minimize(nll, p0, jac=True, bounds=bounds, callback=callback,
                        options={'ftol': 1e-20, 'gtol': 1e-12, 'factr': 10., 'disp':True, 'iprint': 1, 'maxcor': 20})

        stamp.residual = stamp.pixel_values.flatten()
        scene.set_params(result.x)
        resid, partials = model_image(scene.params, scene.sources, stamp)
        dim = stamp.pixel_values
        rim = resid.reshape(stamp.nx, stamp.ny)
        
        fig, axes = pl.subplots(1, 3, sharex=True, sharey=True, figsize=(13.75, 4.25))
        images = [dim, dim-rim, rim]
        labels = ['Data', 'Model', 'Data-Model']
        for k, ax in enumerate(axes):
            c = ax.imshow(images[k].T, origin='lower')
            pl.colorbar(c, ax=ax)
            ax.set_title(labels[k])
        
