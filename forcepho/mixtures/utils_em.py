#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fit a PSF image with a mixture of gaussians using EM
"""

import numpy as np
try:
    import matplotlib.pyplot as pl
    from matplotlib.cm import get_cmap
    from matplotlib.backends.backend_pdf import PdfPages
except(ImportError):
    pass

try:
    import cPickle as pickle
except(ImportError):
    import pickle
from astropy.io import fits
import h5py, json

from .psf_mix_em import fit_mvn_mix
from ..slow.psf import params_to_gauss


__all__ = ["psf_mixture"]


def radial_profile(answer, ax, center):
    pass


def psf_mixture(psfimname, band, nmix, nrepeat=5,
                newstyle=False, width=None, oversample=1):
    """Fit a JWST PSF with a gaussian mixture using EM and make pretty plots of
    the results.  Using `start` and `stop` the fit can be made to subsections
    of the psfimage, which will be extracted as:
    psfdata[start:stop, start:stop]

    :param psfimname:
        Path to the FITS file containing the PSF

    :param band:
        Name of the band (e.g. 'f090w') used for writin output.

    :param nmix:
        Number of gaussian mixtures

    :param start: (optional, default 0.)
        index of the first pixel to use for a subsection of the full psf

    :param stop: (optional, default None)
       index of the last pixel to use for a subsection.  If None, then go to
       the end of the image.

    :param nrepeat:
        The number of initial conditions, a separate fit will be run and saved
        for each set of initial values.

    :returns answer:
       A dictionary keyed by `nmix` containing the fitted parameters as well as
       model and actual images.
    """

    # read in the psf, slice it, and normalize it
    data = np.array(fits.getdata(psfimname))
    hdr = fits.getheader(psfimname)
    size = data.shape
    total_flux = data.sum()
    # where integers refer to the center of a pixel
    center_full = (np.array(size)-1) / 2.
    if width is not None:
        start = np.clip((center_full - width), [0,0], size).astype(int)
        stop = np.clip(center_full + width, [0,0], size).astype(int)
    else:
        start = np.zeros(2).astype(int)
        stop = size
    data = data[start[0]:stop[0], start[1]:stop[1]]
    fractional_flux = data.sum() * 1.0 / total_flux
    data /= data.sum()

    try:
        from astropy.wcs import WCS
        wcs = WCS(hdr)
        plate_scale = np.mean(np.abs(np.linalg.eigvals(3600 * wcs.wcs.cd)))
    except:
        plate_scale = -1

    # --- Do the fit ---
    results = fit_mvn_mix(data, nmix, method_opt='em', method_init='random',
                          repeat=nrepeat, returnfull=True, dlnlike_thresh=1e-9)

    # --- Plotting -----
    outroot = 'gmpsf_{}_ng{}'.format(band, nmix)
    plot_model(results, outroot, log_resid=False)

    if not newstyle:
        return {nmix: results}

    # Make some useful arrays and lists.
    cols = ["x", "y", "vxx", "vyy", "vxy", "amp"]
    dtype = np.dtype([(c, np.float32) for c in cols])
    parameters = np.zeros([nrepeat, nmix], dtype=dtype)
    models, lnlike = [], []
    for i, r in enumerate(results):
        # Note this flips the x and y axes from what is in `result`
        # It also makes sure x and y are relative to the center in the full image
        pars = params_to_gauss(r, oversample=oversample, start=start, center=center_full)
        x, y, vx, vy, vxy, amp = pars
        for j in range(nmix):
            parameters[i, j] = (x[j], y[j], vx[j], vy[j], vxy[j], amp[j])
        # transpose to match the x, y order I expect and the parameters structure
        models.append(r["recon_image"].T)
        lnlike.append(r["final_log_likelihood"])

    with h5py.File(outroot + '.h5', "w") as out:

        out.attrs["nmix"] = nmix
        out.attrs["nrepeat"] = nrepeat
        out.attrs["image_path"] = psfimname
        out.attrs["band"] = band
        out.attrs["start"] = start
        out.attrs["stop"] = stop
        out.attrs["center"] = center_full
        out.attrs["oversample"] = oversample
        out.attrs["flux_fraction"] = fractional_flux

        im = out.create_dataset("psf_image", data=data.T)
        #im.attrs["header"] = json.dumps(hdr)
        im.attrs["arcsec_per_pixel"] = plate_scale
        out.create_dataset("models", data=np.array(models))
        params = out.create_dataset("parameters", data=parameters)
        params.attrs["arcsec_per_pixel"] = oversample * plate_scale
        params.attrs["units"] = "pixels"
        out.create_dataset("lnlike", data=np.array(lnlike))

    output = {"data": data.T, "header": hdr,"flux_fraction": fractional_flux,
              "models": models, "parameters": parameters, "lnlike": lnlike,
              "center": center_full, "start": start, "stop": stop}
    return output


def plot_model(results, outroot, log_resid=True, cmap="viridis"):
    # set up the gaussian colorbar
    gcmap = get_cmap('viridis')
    Z = [[0,0],[0,0]]
    levels = np.arange(0, 0.6, 0.1)
    dummy = pl.contourf(Z, levels, cmap=gcmap)

    pdf = PdfPages(outroot + '.pdf')
    for j, result in enumerate(results):
        fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)
        ax = axes[0, 0]
        d = ax.imshow(result['original_image'], origin='lower', cmap=cmap)
        fig.colorbar(d, ax=ax)
        ax.text(0.1, 0.9, 'Truth', transform=ax.transAxes)

        ax = axes[0, 1]
        m1 = ax.imshow(result['recon_image'], origin='lower', cmap=cmap)
        fig.colorbar(m1, ax=ax)
        ax.text(0.1, 0.9, 'Model', transform=ax.transAxes)

        ax = axes[1, 0]
        # plot log of ratio?
        if log_resid:
            r = ax.imshow(np.log10((result['original_image'] / result['recon_image'])),
                          origin='lower', vmin=-0.5, vmax=0.5, cmap="coolwarm")
        else:
            r = ax.imshow((result['original_image'] - result['recon_image']), 
                          origin='lower', cmap=cmap)
        fig.colorbar(r, ax=ax)
        ax.text(0.1, 0.9, 'Residual', transform=ax.transAxes)
        gax = axes[1, 1]

        gax, amps = draw_ellipses(result, gax, cmap=gcmap)
        pl.colorbar(dummy, ax=gax)
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()


def draw_ellipses(answer, ax, cmap=get_cmap('viridis')):
    from matplotlib.patches import Ellipse

    params = answer['fitted_params'].copy()
    ngauss = int(len(params) / 6)
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
