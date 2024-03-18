#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os, glob, sys
import argparse
import logging

import numpy as np
from astropy.io import fits
import h5py
import arviz # this is required for the plots

from forcepho.mixtures.utils_hmc import Image, infer
from forcepho.mixtures.utils_hmc import display, ee_plot, radial_plot, draw_ellipses
from forcepho.mixtures.psf_mix_hmc import psf_model, psf_prediction


__all__ = ["read_image", "fit_image",
           "convert_psf_data", "package_output", "make_psfstore"]


def read_image(filename, pad=10, ext=0):
    """Read a FITS image into flattenend 1-d array

    Also note that the source is centered *between* the pixels for even size
    arrays, and at the center of a pixel for odd-size arrays. In the inference
    code the convention is that the center of a pixel is at the integer value of
    that pixel.

    Parameters
    ----------
    pad : int
        remove this many rows or columns from each edge of the image before fitting.

    """
    # NOTE: Image is transposed here! so x is the leading dimension
    data = (fits.getdata(filename, ext=ext).T).astype(np.float32)
    slx, sly = slice(pad, data.shape[0] - pad), slice(pad, data.shape[1] - pad)
    data = data[slx, sly]

    nx, ny = data.shape
    cx, cy = (nx - 1) / 2., (ny-1) / 2.
    ypix, xpix = np.meshgrid(np.arange(ny), np.arange(nx))
    xpix, ypix = xpix.flatten(), ypix.flatten()
    data = data.flatten()
    #unc = data.max() / snr
    unc = 0.0
    im = Image(xpix, ypix, data, unc, nx, ny, cx, cy)

    hdr = fits.getheader(filename, ext=ext)

    return im, hdr


def fit_image(image, args, a=None, oversample=1, **kwargs):
    """This wrapper on infer() that computes some required
    input parameters based on image and arguments

    Parameters
    ----------
    image : NamedTuple
        Image data to use in fitting, see forcepho.mixtures.utils_hmc.Image

    args : Namespace
        Must include
        * fix_amplitude
        * ngauss
        * snr
        * num_warmup
        * num_samples

    a : float, optional
        If supplied, fix the combined amplitude to this value

    oversample : int
        number of PSF image pixels per detector pixel

    Extra Parameters
    ----------------
    kwargs : optional
        These are passed to forcepho.mixtures.utils_hmc.infer
    """
    if (not a):
        a = image.data.sum()
    unc = a / (args.snr * np.sqrt(image.nx * image.ny))
    #unc = np.hypot(unc/10, image.data / args.snr)
    if args.fix_amplitude:
        kwargs["afix"] = a
    else:
        kwargs["amax"] = a * 4
    if kwargs.get("ngauss_neg", 0) > 0:
        kwargs["amin"] = a * 3.5
    if args.fit_bg:
        kwargs["maxbg"] = np.abs(np.median(image.data))

    # let some big ole gaussians in
    kwargs["smax"] = np.linspace(5, 15, args.ngauss) * oversample
    # make the bigger gaussians rounder
    kwargs["rho_max"] = np.linspace(0.5, 0.1, args.ngauss)
    # keep teeny-tiny gaussians out.  This sets Gaussians to
    # have FWHM >~ 1 science pixels
    kwargs["smin"] = oversample * 0.4

    best, samples, mcmc = infer(psf_model, image=image.data,
                                xpix=image.xpix, ypix=image.ypix,
                                ngauss=args.ngauss, unc=unc,
                                dense_mass=True,
                                num_warmup=args.num_warmup,
                                num_samples=args.num_samples,
                                **kwargs)

    if args.fix_amplitude:
        best["a"] = a

    return best, samples, mcmc, unc


def rectify(best, image):
    """Generate model image, and best fit parameters accounting for possible
    negative gaussians.
    """
    from copy import deepcopy
    # get neg gaussian params
    best_p, best_m = deepcopy(best), {}
    for k in best_p.keys():
        if k[-2:] == "_m":
            best_m[k[:-2]] = best.pop(k)

    # predict postive gaussians
    model = psf_prediction(image.xpix, image.ypix, **best)

    # predict and subtract negative gaussian
    # also add negative gaussian parameters
    if len(best_m):
        model_m = psf_prediction(image.xpix, image.ypix, **best_m)
        model -= model_m
        for p in best_m.keys():
            v = best_m[p]
            if p == "weight":
                v *= -1
                best[p] *= best["a"]
                best["a"] = 1.0
            best[p] = np.concatenate([best[p], v])

    return best, model


def convert_psf_data(best, cx, cy, nloc=1, nradii=9, oversample=1):
    """Convert numpyro Gaussian mixture model parameters to a parameter
    dictionary close to forcepho format.

    Parameters
    ----------
    best : dict
        Dictionary of psf Gaussian parameters, which are vectors of length
        `ngauss` when appropriate.

    cx : float
        Center of PSF, in PSF image pixels

    cy : float
        Center of PSF, in PSF image pixels

    oversample : int
        Number of pixels in the PSF image per pixel of the science image. So if
        the PSF image if oversampled by a factor of 2, then this number is 2.

    nloc : int
        Make `nloc` copies of the PSF parameter arrays - this will be the first
        dimesion of the output.

    nradii : int
        Make `nradii` copies of the PSF parameter arrays for each location; the
        "sersic_bin" field will be set to an integer corresponding to the index
        of the copy.  This will be the second dimension of the output.

    Returns
    -------
    pars : structured ndarray of shape (nloc, nradii, ngauss)
        Units of the Gaussian parameters are science image pixels.
    """
    ngauss = len(best["x"])
    scale_factor = 1. / oversample

    # This order is important
    cols = ["amp", "xcr", "ycr", "Cxx", "Cyy", "Cxy"]
    pdt = np.dtype([(c, np.float32) for c in cols] +
                   [("sersic_bin", np.int32)])
    pars = np.zeros([nloc, nradii, ngauss], dtype=pdt)

    o = np.argsort(np.array(best["weight"]))[::-1]
    scale = best.get("a", 1.0)
    pars["amp"] = best["weight"][o] * scale
    pars["xcr"] = (best["x"][o] - cx) * scale_factor
    pars["ycr"] = (best["y"][o] - cy) * scale_factor
    sx = (best["sx"][o] * scale_factor)
    if "sy" in best:
        sy = (best["sy"][o] * scale_factor)
    elif "q" in best:
        sy = sx * best["q"][o]
    pars["Cxx"] = sx**2
    pars["Cyy"] = sy**2
    cxy = sx * sy * best["rho"][o]
    pars["Cxy"] = cxy
    pars["sersic_bin"] = np.arange(nradii)[None, :, None]

    return pars, np.float64(scale)


def package_output(h5file, band, pars, pixel_scale=None,
                   oversample=1, image=None, model=None,
                   hdr=None, overwrite=False):
    """
    """
    with h5py.File(h5file, "a") as h5:
        try:
            bg = h5.create_group(band)
        except(ValueError):
            if overwrite:
                del h5[band]
                bg = h5.create_group(band)
        bg.create_dataset("parameters", data=pars.reshape(pars.shape[0], -1))
        bg.attrs["n_psf_per_source"] = pars.shape[1] * pars.shape[2]
        if pixel_scale:
            bg.attrs["arcsec_per_pixel"] = pixel_scale
        bg.attrs["date-added"] = time.strftime("%y%b%d-%H.%M", time.localtime())
        if image is not None:
            im = image.data.reshape(image.nx, image.ny)
            target = bg.create_group("target")
            target.create_dataset("truth", data=image.data)
            target.create_dataset("xpix", data=image.xpix)
            target.create_dataset("ypix", data=image.ypix)
            target.attrs["nx"] = image.nx
            target.attrs["ny"] = image.ny
            target.attrs["cx"] = image.cx
            target.attrs["cy"] = image.cy
            target.attrs["oversample"] = oversample
            #target.attrs["flux_scale"] = scale
        if model is not None:
            mdat = bg.create_group("model")
            m = model.reshape(image.nx, image.ny)
            mdat.create_dataset("model", data=model)
            mdat.attrs["oversample"] = oversample
        if hdr is not None:
            try:
                bg.attrs["psfimage_header"] = hdr.to_string()
            except:
                pass


def combine_psfdatasets(in_files, out_file):
    """
    Parameters
    ----------
    in_files : list of strings
        Paths to the files to combine

    out_file : string
        Location of output combined file
    """
    with h5py.File(out_file, "w") as fdest:
        fdest.attrs["inputs"] = in_files
        for in_file in in_files:
            with h5py.File(in_file, "r") as fsrc:
                for band in fsrc.keys():
                    fsrc.copy(fsrc[band], fdest, band.upper())
                    fdest[band.upper()].attrs["source"] = in_file


def fitsify_output(fn, pars, image, model=None, oversample=1,
                   band="F090W", **hdr_kwargs):

    n = 1
    im = image.data.reshape(image.nx, image.ny)
    pri = fits.PrimaryHDU()
    pri.header["FILTER"] = band.upper()
    hdulist = fits.HDUList(hdus=[pri, fits.ImageHDU(im.T)])
    pri.header[f"EXT{n}"] = "True PSF"
    n += 1
    if model is not None:
        m = model.reshape(image.nx, image.ny)
        hdulist += [fits.ImageHDU(m.T)]
        pri.header[f"EXT{n}"] = "Model PSF"
        n += 1

    hdulist += [fits.BinTableHDU(pars)]
    pri.header[f"EXT{n}"] = "Table of Gaussian Parameters"

    pri.header["SOURCEX"] = image.cx
    pri.header["SOURCEY"] = image.cy
    pri.header["DET_SAMP"] = oversample
    for k, v in hdr_kwargs.items():
        pri.header[k] = v

    hdulist.writeto(fn, overwrite=True)
    return hdulist


def scale_psf_data(pars, cx=0, cy=0, scale_factor=1):
    """change the effective pixel scale for a gaussian parameter dictionary.
    """
    best = {}
    best["weight"] = pars["amp"]
    best["x"] = pars["xcr"] / scale_factor + cx
    best["y"] = pars["ycr"] / scale_factor + cy
    sx = np.sqrt(pars["Cxx"])
    sy = np.sqrt(pars["Cyy"])
    rho = pars["Cxy"] / sx / sy
    best["sx"] = sx / scale_factor
    best["sy"] = sy / scale_factor
    best["rho"] = rho

    return best


if __name__ == "__main__":

    acs_bands = ["f435w", "f606w", "f775w", "f814w", "f850lp", ]
    wfc3_ir_bands = ["f098m", "f105w", "f125w", "f140w", "f160w"]
    wfc3_uvis_bands = ["f225w", "f275w", "f336w"]
    irac_bands = ["irac1", "irac2"]
    nircam_sw_bands = ["f070w", "f090w", "f115w", "f150w", "f200w"]
    nircam_lw_bands = ["f277w", "f335m", "f356w", "f410m", "f444w"]
    nircam_swm_bands = ["f162m", "f182m", "f210m"]
    nircam_lwm_bands = ["f250m", "f300m", "f430m", "f460m", "f480m"]

    parser = argparse.ArgumentParser()
    # output
    parser.add_argument("--outname", type=str, default="./mixtures/psf_nircam_ng4m0.h5")
    parser.add_argument("--fitsout", type=bool, default=0)
    parser.add_argument("--overwrite", type=bool, default=1)
    # band and input
    parser.add_argument("--band", type=str, default="f090w")
    parser.add_argument("--psf_image_name", type=str, default="./psf_images/jwst/f090w_psf.fits")
    parser.add_argument("--oversample", type=int, default=None)
    parser.add_argument("--sci_pix_scale", type=float, default=None, help="for informatioal purposes only")
    parser.add_argument("--trim", type=int, default=0)
    # model
    parser.add_argument("--snr", type=int, default=100)
    parser.add_argument("--ngauss", type=int, default=3)
    parser.add_argument("--ngauss_neg", type=int, default=0)
    parser.add_argument("--fix_amplitude", type=int, default=0)
    parser.add_argument("--fit_bg", type=int, default=0)
    # fitting
    parser.add_argument("--num_warmup", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=512)
    # display
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--savefig", action="store_true")
    args = parser.parse_args()

    #logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('psf-maker')
    logger.setLevel(logging.DEBUG)

    if args.outname:
        outdir = os.path.dirname(args.outname)
        os.makedirs(outdir, exist_ok=True)
    args.ext = 0
    args.pad = args.trim
    band = args.band

    # -- Get the image ---
    filename = args.psf_image_name
    image, hdr = read_image(filename, pad=args.pad, ext=args.ext)
    pixel_scale = args.sci_pix_scale
    if getattr(args, "oversample", None) is None:
        args.oversample = hdr.get("DET_SAMP", 1)
    logger.info(f"Using PSF image {filename} with oversampling by {args.oversample}, "
                f"for a science image pixel scale of {pixel_scale} arcsec/pixel.")
    logger.info(f"PSF image has dimensions {image.nx} x {image.ny} pixels.")

    # --- Do the fit ---
    best, samples, mcmc, unc = fit_image(image, args, a=1.0,
                                         oversample=args.oversample,
                                         dcen=args.oversample / 8,
                                         ngauss_neg=args.ngauss_neg,
                                         max_tree_depth=9)
    best, model = rectify(best, image)
    chi = (model - image.data) / image.data.max()

    # --- save the output ---
    if args.outname:
        outname = args.outname
        pars, scale = convert_psf_data(best, image.cx, image.cy,
                                       nloc=1, nradii=9,
                                       oversample=args.oversample)
        package_output(outname, band.upper(), pars, overwrite=args.overwrite,
                       pixel_scale=pixel_scale, image=None, model=None, hdr=hdr,
                       oversample=args.oversample)
        if args.fitsout:
            ff = fitsify_output(outname.replace(".h5", f"_{band}.fits"),
                                pars[0, 0], image=image, model=model,
                                oversample=args.oversample, flxscl=scale,
                                CDELT1=pixel_scale/3600, CDELT2=pixel_scale/3600.,
                                psfim=os.path.basename(filename))

    # --- Make Figures ---
    if args.show:
        import matplotlib.pyplot as pl
        pl.ion()

    if args.show or args.savefig:
        fig, axes = display(model, image)
        title = r"error as % of peak:"
        title += r"min={:.2f}%, max={:.2f}%".format(chi.min()*100, chi.max()*100)
        title += "\nSummed difference: "
        title += r"{:.2f}%".format(100 * (model - image.data).sum()/image.data.sum())
        fig.suptitle(title)

        import arviz as az
        data = az.from_numpyro(mcmc)
        azax = az.plot_trace(data, compact=True)
        azfig = azax.ravel()[0].figure

        rfig, raxes = radial_plot(image, model, times_r=False)
        d = image.data
        raxes[0].set_ylim(d[d > 0].min(), d.max()*1.1)
        if unc.shape == ():
            raxes[0].axhline(unc)
        import matplotlib.pyplot as pl
        efig, eax = pl.subplots(figsize=(5, 5))
        eax = draw_ellipses(best, eax)
        eax.set_xlim(0, image.nx)
        eax.set_ylim(0, image.ny)
        eax.grid()

        nfig, nax = ee_plot(image, model)

        #mcmc.print_summary()
        logger.info(title)
        logger.info(f"Divergences: {mcmc.get_extra_fields('diverging')['diverging'].sum()}")

    if args.savefig:
        outname = args.outname.replace(".h5", f"_{band}.pdf")
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(outname) as pdf:
            pdf.savefig(fig)
            pdf.savefig(rfig)
            pdf.savefig(nfig)
            pdf.savefig(efig)
            pdf.savefig(azfig)
        pl.close('all')
