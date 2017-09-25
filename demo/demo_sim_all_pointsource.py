# ----------
# Script to fit many point sources in a single image using multiple postage stamps
#-----------

import sys

from copy import deepcopy
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import astropy.io.fits as fits
from astropy import wcs

from forcepho import psf as pointspread
from forcepho.gaussmodel import PostageStamp, Star
from demo_utils import Scene, negative_lnlike_stamp, negative_lnlike_nograd, make_image

       


def make_stamp(imname, center=(None, None), size=(None, None),
               center_type='pixels', psfname=None, fwhm=1.0):
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


def fit_source(ra=53.115325, dec=-27.803518, imname='', psfname=None,
               stamp_size=(100, 100), err_expand=1.0):
    """
    """
    # --- Build the postage stamp ----
    stamp = make_stamp(imname, (ra, dec), stamp_size,
                       psfname=psfname, center_type='celestial')
    stamp.snr = stamp.pixel_values * stamp.ierr
    #err_expand = stamp.snr.max() / 0.3
    stamp.ierr = stamp.ierr.flatten() / err_expand
    
    # override the WCS so coordinates are in pixels
    # The distortion matrix D
    stamp.distortion = np.eye(2)
    # The sky coordinates of the reference pixel
    stamp.crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    stamp.crpix = np.zeros([2])

    # Rotate the PSF by 180 degrees
    T = -1.0 * np.eye(2)
    stamp.psf.covariances = np.matmul(T, np.matmul(stamp.psf.covariances, T.T))
    stamp.psf.means = np.matmul(stamp.psf.means, T)

    # --- get the Scene ---
    scene = Scene()
    sources = [Star()]
    scene.sources = sources

    # ---- Optimization ------
    nll = argfix(negative_lnlike_stamp, scene=scene, stamp=stamp)

    if True:
        def callback(x):
            #nf += 1
            print(x, nll(x))
        callback = None

        # Initial and bounds
        p0 = np.array([stamp.pixel_values.sum(), stamp.nx/2, stamp.ny/2])
        init = p0.copy()
        bounds = [(1, 1e4), (0., stamp_size[0]), (0, stamp_size[1])]
        bounds = None
        
        # Optimize
        from scipy.optimize import minimize
        lbfgsb_opt = {'ftol': 1e-20, 'gtol': 1e-12, 'disp':True, 'iprint': -1, 'maxcor': 20}
        result = minimize(nll, p0, jac=True, bounds=None, callback=callback,
                          options=lbfgsb_opt)

        # plot results
        resid, partials = make_image(result.x, scene, stamp)
        dim = stamp.pixel_values
        mim = resid
        chi = (dim - mim) * stamp.ierr.reshape(stamp.nx, stamp.ny)
        
        fig, axes = pl.subplots(1, 4, sharex=True, sharey=True, figsize=(14.75, 3.25))
        images = [dim, mim, dim-mim, chi]
        labels = ['Data', 'Model', 'Data-Model', '$\chi$']
        for k, ax in enumerate(axes):
            c = ax.imshow(images[k].T, origin='lower')
            pl.colorbar(c, ax=ax)
            ax.set_title(labels[k])

        return result, (fig, axes), nll(result.x), stamp, scene


if __name__ == "__main__":

    imname = '/Users/bjohnson/Projects/nircam/mocks/image/star/sim_cube_F090W_487_001.slp.fits'
    psfname = '/Users/bjohnson/Codes/image/forcepho/data/psf_mixtures/f090_ng6_em_random.p'
    catname = '/Users/bjohnson/Projects/nircam/mocks/image/star/stars_f090w.cat'

    # ---- Read the input catalog -----
    dt = np.dtype([(n, np.float) for n in ['ra', 'dec', 'x', 'y', 'mag', 'counts', 'flux1', 'flux2']])
    cat = np.genfromtxt(catname, usecols=np.arange(1, 9), dtype=dt)
    #cat = cat[:100]

    # --- setup output ---
    out = open('output_pointsource.dat', 'w')
    strfmt = "{}  {:11.5f}   {:11.5f}  {:10.2f}  {:10.2f}  {:14.6f}   {} \n"
    out.write("# i    x    y   flux    counts    chi2/2   nfev\n")
    pdf = PdfPages('pointsource_resid.pdf')

    # ---- loop over things -----
    size = np.array([100, 100])
    all_results, all_pos = [], []
    for i, c in enumerate(cat):
        blob = fit_source(ra=c['ra'], dec=c['dec'],
                          imname=imname, psfname=psfname,
                          stamp_size=size)

        result, (fig, axes), vals, stamp, scene = blob
        all_results.append(result)

        fig.suptitle('obj={}, counts={} x={}, y={}'.format(i, c['counts'], c['x'], c['y']))
        [ax.set_xlim(35, 65) for ax in axes.flat]
        [ax.set_ylim(35, 65) for ax in axes.flat]
        pdf.savefig(fig)
        pl.close(fig)
        
        counts = result.x[0]
        center = np.array(stamp.pixcenter_in_full)
        lo = (center - 0.5 * size).astype(int)
        x, y = lo + result.x[1:]
        all_pos.append((x, y))
        
        print(result.x)
        print(c['counts'])
        
        out.write(strfmt.format(i, x, y, counts, result.fun, stamp.pixel_values.sum(), result.nfev))

    
        
    pdf.close()
    out.close()

    
    xx = np.array([r.x for r in all_results])
    chi2 = np.array([a.fun for a in all_results])
    pos = np.array(all_pos)
