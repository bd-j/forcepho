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
from demo_utils import Scene, negative_lnlike_stamp, negative_lnlike_nograd, chi_vector, make_image


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
    stamp.distortion = np.linalg.inv(distortion)
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
               stamp_size=(100, 100), use_grad=True,
               err_expand=1.0, jitter=0.0, gain=np.inf):
    """
    """
    # --- Build the postage stamp ----
    stamp = make_stamp(imname, (ra, dec), stamp_size,
                       psfname=psfname, center_type='celestial')
    stamp.snr = stamp.pixel_values * stamp.ierr
    stamp.ierr = stamp.ierr.flatten() / err_expand
    counts = stamp.pixel_values.flatten() - stamp.pixel_values.min()
    stamp.ierr = 1.0 / np.sqrt(1/stamp.ierr**2 + jitter**2 + counts/gain)

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
    if use_grad:
        nll = argfix(negative_lnlike_stamp, scene=scene, stamp=stamp)
    else:
        nll = argfix(negative_lnlike_nograd, scene=scene, stamp=stamp)
    if False:
        nll = argfix(chi_vector, scene=scene, stamp=stamp)
        method = 'lm'
        use_grad = False
        
    if True:
        def callback(x):
            #nf += 1
            print(x, nll(x))
        callback = None

        # Initial and bounds
        p0 = np.array([stamp.pixel_values.sum(), stamp.nx/2, stamp.ny/2])
        p0 += np.random.normal(0., [0.1 * p0[0], 0.5, 0.5])
        bounds = [(1, 1e4), (0., stamp_size[0]), (0, stamp_size[1])]
        bounds = None
        
        # Optimize
        from scipy.optimize import minimize
        lbfgsb_opt = {'ftol': 1e-20, 'gtol': 1e-12, 'disp':True, 'iprint': -1, 'maxcor': 20}
        result = minimize(nll, p0, jac=use_grad, bounds=None, callback=callback,
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
    cat = cat[:100]

    # --- setup output ---
    use_grad = True
    fn = 'output_pointsource.dat'
    pn = 'pointsource_resid.pdf'
    out = open(fn, 'w')
    strfmt = "{}  {:11.5f}   {:11.5f}  {:10.2f}  {:10.2f}  {:14.6f}   {} \n"
    dt2 = np.dtype([(n, np.float) for n in ['id', 'x', 'y', 'counts', 'halfchisq', 'sum', 'nfev']])
    out.write("# i    x    y   flux    counts    chi2/2   nfev\n")
    pdf = PdfPages(pn)

    # ---- loop over things -----
    size = np.array([30, 30])
    all_results, all_pos = [], []
    for i, c in enumerate(cat):
        blob = fit_source(ra=c['ra'], dec=c['dec'],
                          imname=imname, psfname=psfname,
                          stamp_size=size,
                          err_expand=1, jitter=0.5, gain=2.0,
                          use_grad=use_grad)

        result, (fig, axes), vals, stamp, scene = blob
        all_results.append(result)

        axes[0].text(0.05, 0.82,
                     'obj={}\ncounts={:5.1f}\nx={:4.2f}, y={:4.2f}'.format(i+1, c['counts'], c['x'], c['y']),
                     transform=axes[0].transAxes, size=10)
        #[ax.set_xlim(35, 65) for ax in axes.flat]
        #[ax.set_ylim(35, 65) for ax in axes.flat]
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

    ocat = np.genfromtxt(fn, dtype=dt2)
    icat = cat
    ratio = (ocat['counts'] / icat['counts'])
    
    fig, ax = pl.subplots()
    ax.plot(icat['counts'], ratio, 'o')
    ax.set_xscale('log')
    ax.set_xlabel('Input counts')
    ax.set_ylabel('Output/Input')
    ax.axhline(ratio.mean(), label='$\mu={:4.2f}$'.format(ratio.mean()), color='k', linestyle='--')
    ax.axhline(ratio.mean() + ratio.std(), label='$\mu+/-\sigma, \sigma={:3.3f}$'.format(ratio.std()), color='k', linestyle=':')
    ax.axhline(ratio.mean() - ratio.std(), color='k', linestyle=':')
    ax.legend()


    fig, ax = pl.subplots()
    dx = ocat['x'] - icat['x'] + 1.0
    dy = ocat['y'] - icat['y'] + 1.0
    ax.plot(dx, dy, 'o')
    ax.set_xlabel('$\Delta x$')
    ax.set_ylabel('$\Delta y$')
    txt = '$\mu_x, \sigma_x=${:3.2f}, {:3.2f}\n$\mu_y, \sigma_y=${:3.2f}, {:3.2f}'
    ax.text(0.1, 0.9, txt.format(dx.mean(), dx.std(), dy.mean(), dy.std()),
            transform=ax.transAxes)

    vals = icat['mag'], ratio - ratio.mean(), dy - dy.mean(), dx - dx.mean()
    labs = ['mag', 'fout/fin', '$\Delta y$', '$\Delta x$']
    fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)
    xoff, yoff = icat['x'] % 1, icat['y'] % 1
    for ax, v, l in zip(axes.flatten(), vals, labs):
        c = ax.scatter(xoff, yoff, c=v, cmap='RdBu')
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_title(l)
    #ax.scatter(dx, dy, c=ratio)
        ax.set_xlabel('$\delta x$')
        ax.set_ylabel('$\delta y$')

    fig, ax = pl.subplots()
    c = ax.scatter(dx, dy, c=ratio-ratio.mean(), cmap='RdBu')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_title('fout/fin')

    #ax.imshow((stamp.ierr.reshape(stamp.nx, stamp.ny) * stamp.pixel_values).T, origin='lower')
    
    pl.show()
