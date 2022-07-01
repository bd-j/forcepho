#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

import matplotlib.pyplot as pl
from astropy.io import fits


from test_utils import get_parser, get_grid_params
from test_plot import plot_trace, plot_corner, plot_residual
from test_plot import make_catalog, compare_parameters, compare_apflux

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def make_tag(config):
    # this could be programmitic
    tag = f"sersic{config.sersic[0]:.1f}_rhalf{config.rhalf[0]:.3f}_q{config.q[0]:01.2f}"
    tag += f"_band{config.bands[0]}_snr{config.snr:03.0f}_noise{config.add_noise:.0f}"
    return tag


def make_all_tags(grid, config):
    from argparse import Namespace
    pars = Namespace(add_noise=config.add_noise,
                     bands=config.bands)
    tags = []
    for row in grid:
        [setattr(pars, p, [row[p]]) for p in ["sersic", "rhalf", "q"]]
        [setattr(pars, p, row[p]) for p in ["snr"]]
        tags.append(make_tag(pars))
    return tags



if __name__ == "__main__":

    print(f"HASGPU={HASGPU}")

    # ------------------
    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        sigma_psf=[2.5],
                        rhalf=[0.2],
                        sersic=[2.0],
                        psfstore="./psf_hlf_ng4.h5")
    parser.add_argument("--dir", type=str, default="./output/hst/")
    parser.add_argument("--test_grid", type=str, default="./test_psf_grid.yml")
    parser.add_argument("--start", type=int, default=0)
    # filter/psf
    parser.add_argument("--bandname", type=str, default="F435W")
    config = parser.parse_args()

    thisband = config.bandname
    config.bands = [thisband.upper()]
    config.banddir = os.path.join(config.dir, config.bandname)

    print(os.path.join(config.banddir, "*grid*.fits"))
    outgridname = glob.glob(os.path.join(config.banddir, "*grid*.fits"))[0]

    # Make summary plots
    tcat = fits.getdata(outgridname)
    tags = make_all_tags(tcat, config)
    tags = [os.path.join(config.banddir, tag, tag) for tag in tags]
    scat = make_catalog(tags, bands=config.bands)
    comp = [("rhalf", "sersic"), ("sersic", "rhalf"), ("q", "rhalf")]
    for show, by in comp:
        fig, axes = pl.subplots(2, 1, figsize=(8, 6))
        fig, ax, cb = compare_parameters(scat, tcat, show, dfax=(fig, axes[0]), colorby=by)
        fig, ax, cb = compare_parameters(scat, tcat, show, dfax=(fig, axes[1]), colorby=by,
                                         as_delta=True, add_jitter=True)
        fig.colorbar(cb, label=by, ax=axes, orientation="vertical")
        fig.savefig(os.path.join(config.banddir, f"{thisband.lower()}_{show}_comparison.png"))
        pl.close(fig)

    fig, ax = compare_apflux(scat, tcat, band=config.bands, colorby="rhalf")
    ax.set_ylim(0.9, 1.1) #for ax in axes]
    fig.savefig(os.path.join(config.banddir, f"{thisband.lower()}_flux_comparison.png"), dpi=200)

