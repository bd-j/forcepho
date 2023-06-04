#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, shutil
import logging
import numpy as np
import json, yaml
import matplotlib.pyplot as pl

from astropy.io import fits
from astropy.wcs import WCS

from forcepho.region import CircularRegion
from forcepho.patches import FITSPatch, CPUPatchMixin
from forcepho.superscene import make_bounds, Galaxy
from forcepho.utils import write_to_disk
from forcepho.fitting import run_lmc

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import write_fits_to

from psf_plot import plot_trace, plot_corner, plot_residual


class Patcher(FITSPatch, CPUPatchMixin):
    pass


def make_tag(config):
    # this could be programmitic
    tag = f"sersic{config.sersic[0]:.1f}_rhalf{config.rhalf[0]:.3f}_q{config.q[0]:01.2f}"
    tag += f"_band{config.bands[0]}_snr{config.snr:03.0f}_noise{config.add_noise:.0f}"
    return tag


def make_image(config):
    """ Makes the image, uncertainty, and scene
    """
    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]
    # make empty stamp and put scene in it
    stamp = make_stamp(band, scale=scale, nx=config.nx, ny=config.ny)
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       q=config.q, pa=config.pa,
                       rhalf=config.rhalf, sersic=config.sersic)
    # Render the scene in galsim, using image based PSF
    psf = get_galsim_psf(scale, psfimage=config.psfimage, sigma_psf=None)
    im = galsim_model(scene, stamp, psf=psf)

    # Noisify
    noise_per_pix = compute_noise_level(scene, config)
    unc = np.ones_like(im)*noise_per_pix
    noise = np.random.normal(0, noise_per_pix, size=im.shape)
    if config.add_noise:
        im += noise

    # write the test image
    hdul, wcs = stamp.to_fits()
    hdr = hdul[0].header
    hdr["FILTER"] = band
    hdr["SNR"] = config.snr
    hdr["NOISED"] = config.add_noise
    hdr["PSF"] = config.psfimage
    write_fits_to(config.image_name, im, unc, hdr, config.bands,
                  noise=noise, scene=scene)
    hdul.close()


def fit_image(config, shape_cols=Galaxy.SHAPE_COLS,
              bounds_kwargs=dict(n_pix=1.5,
                                 rhalf_range=(0.001, 1.0),
                                 sersic_range=(0.8, 6.0))):

    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]

    # build the scene
    cat = fits.getdata(config.image_name, -1)
    bands = fits.getheader(config.image_name, -1)["FILTERS"].split(",")
    active = cat.copy()
    fixed = None
    bounds = make_bounds(active, bands, shapenames=shape_cols, **bounds_kwargs)

    # a region that covers the whole image
    wcs = WCS(fits.getheader(config.image_name, "SCI"))
    ra, dec = wcs.pixel_to_world_values(config.nx/2, config.ny/2)
    radius = np.max(config.nx * np.array(config.scales))  / np.sqrt(2) * 1.1 / 3600.
    region = CircularRegion(float(ra), float(dec), radius)


    # load the image data
    patcher = Patcher(fitsfiles=[config.image_name],
                      sci_ext="SCI", unc_ext="ERR",
                      psfstore=config.psfstore,
                      splinedata=config.splinedatafile,
                      return_residual=True)


    # prepare model and data, and sample
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=shape_cols)
    out, step, stats = run_lmc(model, q.copy(),
                               n_draws=config.sampling_draws,
                               warmup=config.warmup,
                               z_cov=np.eye(len(q)), full=True,
                               weight=max(10, active["n_iter"].min()),
                               discard_tuned_samples=False,
                               max_treedepth=config.max_treedepth,
                               progressbar=config.progressbar)

    # Check results back in and end and write everything to disk
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=0)
    write_to_disk(out, config.outroot, model, config)


if __name__ == "__main__":

    # ------------------
    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        rhalf=[0.1],
                        sersic=[2.0],
                        q=[0.95],
                        pa=0.0,
                        snr=[100.0],
                        add_noise=0,
                        psfstore="./psf_mixture.h5")
    parser.add_argument("--tag", type=str, default="")
    # I/O
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--output_dir", type=str, default="./output/test")
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--psfdir", type=str, default="./psf_images/jwst",
                        help="directory continaing the PSF images as <band>_psf.fits")
    # parameters
    parser.add_argument("--bandname", type=str, default="F200W")
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=1024)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=1)
    config = parser.parse_args()

    # --- decide the band/psf to use ---
    thisband = config.bandname
    config.bands = [thisband.upper()]
    config.psfimage = os.path.join(config.psfdir, f"{thisband.lower()}_psf.fits")

    # make directories and names
    config.tag = make_tag(config)
    config.outdir = config.output_dir
    os.makedirs(config.outdir, exist_ok=True)
    config.outroot = os.path.join(config.outdir, config.tag)
    config.image_name = f"{config.outroot}_data.fits"

    # ---------------------
    # --- Make the data ---
    size_img = int(np.clip(20.0*config.rhalf[0]/config.scales[0], 64, 256))
    config.nx = size_img
    config.ny = size_img
    if not os.path.exists(config.image_name):
        make_image(config)
    else:
        pass

    # --------------------
    # --- Fit the data ---
    fit_image(config)

    # --------------------
    # --- make figures ---
    patchname = f"{config.outroot}_samples.h5"
    title = config.tag.replace("_", ", ")

    tfig, ax = plot_trace(patchname)
    tfig.suptitle(title)
    tfig.tight_layout()
    tfig.savefig(f"{config.outroot}_trace.png", dpi=200)
    pl.close(tfig)

    cfig, caxes = plot_corner(patchname)
    cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
    cfig.savefig(f"{config.outroot}_corner.png", dpi=200)
    pl.close(cfig)

    rfig, raxes, rcb, val = plot_residual(patchname)
    rfig.savefig(f"{config.outroot}_residual.png", dpi=200)
    pl.close(rfig)

