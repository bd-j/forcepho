import numpy as np
import matplotlib.pyplot as pl
from matplotlib.cm import get_cmap

import cPickle as pickle
from astropy.io import fits as pyfits

from gaussian_psf import fit_mvn_mix


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


def params_to_gauss(answer, oversample=8, start=0, center=504):
    """Convert the fitted parameters to the parameters used in the PSF gaussian mixture.

    :returns mux:
        The center x of each gaussian, in detector coordinates (i.e. not PSF
        image pixels, but actual detector pixels)

    :returns muy:
        The center y of each gaussian, in detector coordinates (i.e. not PSF
        image pixels, but actual detector pixels)

    :returns vx:
        The 0,0 entry of the covariance matrix (sigma_x^2) in (detector pixels)^2

    :returns vy:
        The 1,1 entry of the covariance matrix (sigma_y^2) in (detector pixels)^2

    :returns vxy:
        The 1,0 or 0,1 entry of the covariance matrix (rho * sigma_x * sigma_y)
        in (detector pixels)^2

    :returns amp:
    
    """
    params = answer['fitted_params'].copy()
    ngauss = len(params) / 6
    params = params.reshape(ngauss, 6)
    # is this right?
    #TODO: work out zero index vs 0.5 index issues
    # need to flip x and y here
    mu = (params[:,1:3][::-1] + start - center) / oversample
    sy = params[:, 3] / oversample
    sx = params[:, 4] / oversample
    vxy = params[:, 5] * sx * sy
    amp = params[:, 0]

    return mu[:, 0], mu[:, 1], sx**2, sy**2, vxy, amp


if __name__ == "__main__":

    psfname = "/Users/bjohnson/Projects/nircam/mocks/image/psfs/PSF_NIRCam_F090W.fits"

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

    with open('f090_ng6_em_random.p', 'wb') as out:
        pickle.dump(ans_all_em_random, out)
    # --- Plotting -----
    # set up the gaussian colorbar
    gcmap = get_cmap('viridis')
    Z = [[0,0],[0,0]]
    levels = np.arange(0, 0.6, 0.1)
    dummy = pl.contourf(Z, levels, cmap=gcmap)

    # set up the figure
    fig, axes = pl.subplots(nrepeat + 1, 3, sharex=True, sharey=True)
    d = axes[0, 0].imshow(data, origin='lower')
    fig.colorbar(d, ax=axes[0,0])
    axes[0, 1].contour(data, levels=[5e-4, 1e-3, 2e-3], colors='k')
    cbar=fig.colorbar(d, ax=axes[0,1])
    #cbar.clear()

    for i in range(1, nrepeat+1):
        m1 = axes[i, 0].imshow((ans_all_em_random[nmix][i-1]['recon_image']), origin='lower')
        fig.colorbar(m1, ax=axes[i, 0])
        r = axes[i, 1].imshow((data - ans_all_em_random[nmix][i-1]['recon_image']), origin='lower')
        fig.colorbar(r, ax=axes[i, 1])
        gax = axes[i, 2]
        
        gax, amps = draw_ellipses(ans_all_em_random[nmix][i-1], gax, cmap=gcmap)
        pl.colorbar(dummy, ax=gax)
        
