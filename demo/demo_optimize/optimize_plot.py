#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob
import argparse
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter

from forcepho.postprocess import Samples, Residuals
from forcepho.utils.corner import allcorner, scatter, marginal, corner, get_spans, prettify_axes

fsize = 8, 9.5


def plot_corner(samples, smooth=0.05, hkwargs=dict(alpha=0.65),
                dkwargs=dict(color="red", marker="."), fsize=(8, 8)):
    labels = samples.chaincat.dtype.names[1:]

    fig, axes = pl.subplots(7, 7, figsize=fsize, gridspec_kw=dict(wspace=0.08, hspace=0.08))
    axes = allcorner(samples.chain.T, labels, axes,
                     color="royalblue",  # qcolor="black",
                     smooth=smooth, hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)
    for i, ax in enumerate(np.diag(axes)):
        #ax.axvline(truth[0, i], color="red")
        if (labels[i] == "ra") | (labels[i] == "dec"):
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            axes[-1, i].xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ymap = get_map(samples)
    scatter(ymap.T, axes, zorder=20, color="k", marker=".")
    for ax, val in zip(np.diag(axes), ymap[0]):
        ax.axvline(val, linestyle=":", color="k")

    return fig, axes


def plot_residual(patchname, vmin=-1, vmax=5, rfig=None, raxes=None):
    s = Samples(patchname)
    r = Residuals(patchname.replace("samples", "residuals"))
    data, _, _ = r.make_exp(value="data")
    delta, _, _ = r.make_exp(value="residual")
    ierr, _, _ = r.make_exp(value="ierr")
    if raxes is None:
        rfig, raxes = pl.subplots(2, 3, gridspec_kw=dict(height_ratios=[1, 40]))
    kw = dict(origin="lower", vmin=vmin, vmax=vmax)
    cb = raxes[1, 0].imshow((data * ierr).T, **kw)
    cb = raxes[1, 1].imshow((delta * ierr).T, **kw)
    cb = raxes[1, 2].imshow(((data-delta) * ierr).T, **kw)
    [pl.colorbar(cb, label=r"$\chi$", cax=ax, orientation="horizontal")
     for ax in raxes[0, :]]

    val = s.get_sample_cat(-1)
    return rfig, raxes, cb, val


def get_map(s):
    lnp = s.stats["model_logp"]
    ind_ml = np.argmax(lnp)
    #row_map = s.get_sample_cat(ind_ml)[0]
    #ymap = np.atleast_2d([row_map[c] for c in s.bands + s.shape_cols])
    ymap = np.atleast_2d(s.chain[ind_ml, :])

    return ymap


if __name__ == "__main__":

    pl.rcParams["xtick.direction"] = "in"
    pl.rcParams["ytick.direction"] = "in"

    tdir = sys.argv[1]
    patchname = f"{tdir}_samples.h5"
    samples = Samples(patchname)
    q_map = get_map(samples)
    dirname = os.path.dirname(patchname)
    tag = os.path.basename(patchname).replace(".h5", "")
    title = ", ".join(tag.split("_")[:-1])

    cfig, caxes = plot_corner(samples)
    skwargs = dict(zorder=20, s=12)
    scatter(np.atleast_2d(samples.q_initial).T, caxes, color="k", label="Truth", **skwargs)
    scatter(np.atleast_2d(samples.q_start).T, caxes, color="firebrick", label="Random Start", **skwargs)
    scatter(np.atleast_2d(samples.q_postop).T, caxes, color="orange", label="Post-optimization", **skwargs)
    if hasattr(samples, "q_postlinear"):
        scatter(np.atleast_2d(samples.q_postlinear).T, caxes, color="green", label="Post-linear flux optimization", **skwargs)
    scatter(np.atleast_2d(q_map).T, caxes, color="darkslateblue", label="MAP sample", **skwargs)
    for i in range(len(caxes)):
        r = caxes[-1,i].get_xlim()
        caxes[i,i].set_xlim(*r)

    handles, labels = caxes[1, 0].get_legend_handles_labels()
    cfig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.85, 0.8), frameon=True, fontsize=12)
    cfig.text(0.55, 0.8, title, transform=cfig.transFigure)
    cfig.savefig(f"{dirname}/{tag}_corner.png", dpi=200)
    pl.close(cfig)

    sys.exit()

    rfig, raxes, rcb, val = plot_residual(patchname)
    rfig.savefig(f"{dirname}/{tag}_residual.png", dpi=200)
    pl.close(rfig)