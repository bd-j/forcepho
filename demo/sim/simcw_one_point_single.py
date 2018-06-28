# ----------
# Script to fit many point sources in a single Guitarra simulated image using
# multiple postage stamps
#-----------

import sys, os
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import astropy.io.fits as fits
from astropy import wcs

from forcepho import paths
from forcepho import psf as pointspread
from forcepho.gaussmodel import Star
from forcepho.data import PostageStamp
from forcepho.sources import Scene

from demo_utils import make_real_stamp as make_stamp
from demo_utils import make_image, negative_lnlike_stamp, negative_lnlike_nograd


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
    # The scale matrix D
    stamp.scale = np.eye(2)
    # The sky coordinates of the reference pixel
    stamp.crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    stamp.crpix = np.zeros([2])

    # Rotate the PSF by 180 degrees
    T = -1.0 * np.eye(2)
    stamp.psf.covariances = np.matmul(T, np.matmul(stamp.psf.covariances, T.T))
    stamp.psf.means = np.matmul(stamp.psf.means, T)

    # --- get the Scene ---
    scene = Scene(galaxy=False)
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


def plot_residuals(fn='output_pointsource.dat',
                   catname=os.path.join(paths.starsims, 'stars_f090w.cat')):
    
    dt = np.dtype([(n, np.float) for n in ['ra', 'dec', 'x', 'y', 'mag', 'counts', 'flux1', 'flux2']])
    dt2 = np.dtype([(n, np.float) for n in ['id', 'x', 'y', 'counts', 'halfchisq', 'sum', 'nfev']])
    icat = np.genfromtxt(catname, usecols=np.arange(1, 9), dtype=dt)
    icat = icat[:100]
    ocat = np.genfromtxt(fn, dtype=dt2)
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


if __name__ == "__main__":

    imname = os.path.join(paths.starsims, 'sim_cube_F090W_487_001.slp.fits')
    psfname = os.path.join(paths.psfmixtures, 'f090_ng6_em_random.p')
    catname = os.path.join(paths.starsims, 'stars_f090w.cat')

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


