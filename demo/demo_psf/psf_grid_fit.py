#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, shutil
import logging
import numpy as np
import json, yaml
import matplotlib.pyplot as pl
from astropy.io import fits

from forcepho.utils import NumpyEncoder

from demo_utils import get_parser
from psf_one_fit import make_image, fit_image

from psf_plot import plot_trace, plot_corner, plot_residual
from psf_plot import make_catalog, compare_parameters, compare_apflux


def make_tag(config):
    # this could be programmitic
    tag = f"sersic{config.sersic[0]:.1f}_rhalf{config.rhalf[0]:.3f}_q{config.q[0]:01.2f}"
    tag += f"_band{config.bands[0]}_snr{config.snr:03.0f}_noise{config.add_noise:.0f}"
    return tag


def make_grid(grid={}, grid_spec=""):
    from itertools import product

    if "yml" in grid_spec:
        with open(grid_spec, "r") as f:
            grid = yaml.load(f, Loader=yaml.Loader)

    if "fits" in grid_spec:
            params = fits.getdata(grid_spec)
    else:
        names = list(grid.keys())
        names.sort()
        params = list(product(*[grid[k] for k in names]))

        cols = list([(k, np.array(grid[k]).dtype) for k in names])
        dtype = np.dtype(cols)
        params = np.array(params, dtype=dtype)

    return params


if __name__ == "__main__":

    # ------------------
    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        add_noise=0,
                        psfstore="./psf_mixture.h5")
    parser.add_argument("--tag", type=str, default="")
    # I/O
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--psfdir", type=str, default="./psf_images/jwst",
                        help="directory continaing the PSF images as <band>_psf.fits")
    # parameters
    parser.add_argument("--bandname", type=str, default="F200W")
    parser.add_argument("--parameter_grid", type=str, default="./psf_grid.yml")
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
    config.banddir = os.path.join(config.output_dir, config.bandname)

    os.makedirs(config.banddir, exist_ok=True)
    shutil.copy(config.psfimage, os.path.join(config.banddir, os.path.basename(config.psfimage)))
    shutil.copy(config.psfstore, os.path.join(config.banddir, os.path.basename(config.psfstore)))
    # copy the config data
    with open(f"{config.banddir}/config.json", "w") as cfg:
        json.dump(vars(config), cfg, cls=NumpyEncoder)

    # --- Set up the grid ---
    params = make_grid(grid_spec=config.parameter_grid)
    # write the input grid
    gname = config.parameter_grid.replace(".yml", ".fits")
    outgridname = os.path.join(config.banddir, os.path.basename(gname))
    fits.writeto(outgridname, params, overwrite=True)

    # loop over grid, generating images and fitting
    tags = []
    for param in params:
        # set parameters in config
        config.rhalf = [param["rhalf"]]
        config.sersic = [param["sersic"]]
        config.q = [param["q"]]
        config.snr = param["snr"]
        config.pa = 0

        size_img = int(np.clip(20.0*config.rhalf[0]/config.scales[0], 64, 256))
        config.nx = size_img
        config.ny = size_img

        # make directories and names
        config.tag = make_tag(config)
        config.outdir = os.path.join(config.banddir, config.tag)
        os.makedirs(config.outdir, exist_ok=True)
        config.outroot = os.path.join(config.outdir, config.tag)
        config.image_name = f"{config.outroot}_data.fits"

        # ---------------------
        # --- Make the data ---
        if not os.path.exists(config.image_name):
            make_image(config)

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

        tags.append(config.outroot)

