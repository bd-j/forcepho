#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
import h5py
from utils import make_chaincat, Galaxy, make_boundscat


def show_marg(truth, chain, bandlist, axes=None,
              span=0.999999426697, bounds=None,
              truth_kwargs=dict(linestyle="--", color="tomato"),
              post_kwargs=dict(alpha=0.5, color="royalblue")):

    q = np.array([0.5 - 0.5 * span, 0.5 + 0.5 * span])

    cols = bandlist + Galaxy.SHAPE_COLS
    for i, col in enumerate(cols):
        ax = axes.flat[i]
        xx = chain[col]
        lim = np.percentile(xx, list(q * 100.))
        if bounds is not None:
            lim = bounds[col]

        ax.hist(chain[col], range=lim, density=True, **post_kwargs)
        ax.axvline(truth[col], **truth_kwargs)
        ax.set_xlabel(col)
        ax.set_xlim(*lim)
    return ax


def show_chain(truth, chain, bandlist, axes=None,
               span=0.999999426697, bounds=None,
               truth_kwargs=dict(linestyle="--", color="tomato"),
               post_kwargs=dict(alpha=0.5, color="royalblue")):
    q = np.array([0.5 - 0.5 * span, 0.5 + 0.5 * span])
    cols = bandlist + Galaxy.SHAPE_COLS
    for i, col in enumerate(cols):
        ax = axes.flat[i]
        xx = chain[col]
        lim = np.percentile(xx, list(q * 100.))
        if bounds is not None:
            lim = bounds[col]

        ax.plot(chain[col], **post_kwargs)
        ax.set_ylim(*lim)
        ax.axhline(truth[col], **truth_kwargs)
        ax.set_ylabel(col)
    return ax


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--patchID", type=int, default=1)
    parser.add_argument("--patch_dir", type=str, default="../output/run3")
    parser.add_argument("--marginalized", action="store_true")
    parser.add_argument("--n_show", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    original = fits.getdata("../data/galsim_galaxy_grid_cat.fits")
    original["q"] = np.sqrt(original["q"])

    pn = "patch{:04.0f}_results.h5".format(args.patchID)
    fn = os.path.join(args.patch_dir, pn)
    with h5py.File(fn, "r") as res:
        chain = res["chain"][:]
        #bands = res["bandlist"][:].astype("U").tolist()
        bands = ["Fclear"]
        ref = res["reference_coordinates"][:]
        active = res["active"][:]
        lower = res["lower_bound"][:]
        upper = res["upper_bound"][:]

    cat = make_chaincat(chain, bands, active, ref)
    bounds = make_boundscat(lower, upper, bands, active, ref)
    orig = original[active["source_index"]]

    pl.ion()
    start = args.start
    N = args.n_show
    fig, axes = pl.subplots(N, 7)
    for i, truth in enumerate(orig[start:start+N]):
        if args.marginalized:
            show_marg(truth, cat[i+start], bands, axes[i,:], bounds=bounds[i+start])
        else:
            show_chain(truth, cat[i+start], bands, axes[i,:], bounds=bounds[i+start])

    #[ax.set_yticklabels("") for ax in axes.flat]