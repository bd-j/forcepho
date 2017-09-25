import numpy as np
import matplotlib.pyplot as pl
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_pdf import PdfPages

import cPickle as pickle
from astropy.io import fits as pyfits

from .gaussian_psf import fit_mvn_mix
from ..psf import params_to_gauss


def draw_ellipses(answer, ax, cmap=get_cmap('viridis')):
    from matplotlib.patches import Ellipse

    params = answer['fitted_params'].copy()
    ngauss = len(params) / 6
    params = params.reshape(ngauss, 6)
    for i in range(ngauss):
        # need to swap axes here, not sure why
        mu = params[i, 1:3][::-1]
        sy = params[i, 3]
        sx = params[i, 4]
        sxy = params[i, 5] * sx * sy
        # construct covar matrix and get eigenvalues
        S = np.array([[sx**2, sxy],[sxy, sy**2]])
        vals, vecs = np.linalg.eig(S)
        # get ellipse params
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=w, height=h, angle=theta)
        ax.add_artist(ell)
        #e.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ell.set_facecolor(cmap(params[i,0]))

    return ax, params[:, 0]


def radial_profile(answer, ax, center):
    pass
    


if __name__ == "__main__":

    psfname = "/Users/bjohnson/Projects/nircam/mocks/image/psfs/PSF_NIRCam_F090W.fits"
    band = 'f090w'
    # read in the psf and normalize it
    data = np.array(pyfits.getdata(psfname))
    start, stop = 400, 600  # this contains about 88.5% of total flux
    data = data[start:stop, start:stop]
    data /= data.sum()

    # --- Do the fit ---
    nmix = 6
    nrepeat = 5
    ans_all_em_random = {}
    ans_all_em_random[nmix] = fit_mvn_mix(data, nmix, method_opt='em', method_init='random',
                                          repeat=nrepeat, returnfull=True, dlnlike_thresh=1e-9)

    #with open('f090_ng6_em_random.p', 'wb') as out:
    #    pickle.dump(ans_all_em_random, out)
    # --- Plotting -----
    # set up the gaussian colorbar
    gcmap = get_cmap('viridis')
    Z = [[0,0],[0,0]]
    levels = np.arange(0, 0.6, 0.1)
    dummy = pl.contourf(Z, levels, cmap=gcmap)

    # set up the figure
    #fig, axes = pl.subplots(nrepeat + 1, 3, sharex=True, sharey=True)
    #d = axes[0, 0].imshow(data, origin='lower')
    #fig.colorbar(d, ax=axes[0,0])
    #axes[0, 1].contour(data, levels=[5e-4, 1e-3, 2e-3], colors='k')
    #cbar=fig.colorbar(d, ax=axes[0,1])
    #cbar.clear()

    pdf = PdfPages('gmpsf_{}_ng{}.pdf'.format(band, nmix))
    for i in range(1, nrepeat+1):
        fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)
        ax = axes[0, 0]
        d = ax.imshow(data, origin='lower')
        fig.colorbar(d, ax=ax)
        ax.text(0.1, 0.9, 'Truth', transform=ax.transAxes)
        #ax = axes[0, 1]
        #ax.contour(data, levels=[5e-4, 1e-3, 2e-3], colors='k')
        #cbar = fig.colorbar(d, ax=ax)
        #cbar.ax.set_visible(False)
        #axes[0,2].set_visible(False)

        ax = axes[0, 1]
        m1 = ax.imshow((ans_all_em_random[nmix][i-1]['recon_image']), origin='lower')
        fig.colorbar(m1, ax=ax)
        ax.text(0.1, 0.9, 'Model', transform=ax.transAxes)
        ax = axes[1, 0]
        r = ax.imshow((data - ans_all_em_random[nmix][i-1]['recon_image']), origin='lower')
        fig.colorbar(r, ax=ax)
        ax.text(0.1, 0.9, 'Residual', transform=ax.transAxes)
        gax = axes[1, 1]

        gax, amps = draw_ellipses(ans_all_em_random[nmix][i-1], gax, cmap=gcmap)
        pl.colorbar(dummy, ax=gax)
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()
