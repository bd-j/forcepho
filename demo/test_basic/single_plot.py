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
from prospect.plotting.corner import allcorner, scatter, marginal, corner, get_spans, prettify_axes

fsize = 8, 9.5


def plot_trace(patchname, title_fmt=".2g", fsize=fsize):
    samples = Samples(patchname)
    fig, axes = pl.subplots(7, sharex=True, figsize=fsize)
    samples.show_chain(0, axes=np.array(axes), truth=samples.active[0])
    for i, c in enumerate(samples.bands + samples.shape_cols):
        ax = axes[i]
        xx = samples.chaincat[0][c]
        truth = samples.active[c][0]
        lim = np.percentile(xx, [1, 99])
        ax.set_ylim(*lim)
        v = np.percentile(xx, [16, 50, 84])
        qm, qp = np.diff(v)
        p = np.max(np.ceil(np.abs(np.log10(np.diff(v))))) + 1
        # could do better here about automating the format
        cfmt = "{{:.{}g}}".format(int(p)).format
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(cfmt(v[1]), fmt(qm), fmt(qp))
        ax.text(1.0, 0.7, title, color="blue", transform=ax.transAxes)
        ax.text(1.0, 0.2, cfmt(truth), color="red", transform=ax.transAxes)

    axes[-1].set_xlabel("HMC iteration")

    return fig, axes


def plot_corner(patchname, smooth=0.05, hkwargs=dict(alpha=0.65),
                dkwargs=dict(color="red", marker="."), fsize=(8, 8)):
    from prospect.plotting.corner import allcorner, scatter
    samples = Samples(patchname)
    truth = np.atleast_2d(samples.starting_position)
    labels = samples.chaincat.dtype.names[1:]

    fig, axes = pl.subplots(7, 7, figsize=fsize)
    axes = allcorner(samples.chain.T, labels, axes,
                     color="royalblue",  # qcolor="black",
                     psamples=truth.T,
                     smooth=smooth, hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)
    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")
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

    tdir = sys.argv[1]
    patchname = f"{tdir}_samples.h5"
    dirname = os.path.dirname(patchname)
    tag = os.path.basename(patchname).replace(".h5", "")
    title = tag.replace("_", ", ")

    tfig, ax = plot_trace(patchname)
    tfig.suptitle(title)
    tfig.tight_layout()
    tfig.savefig(f"{dirname}/{tag}_trace.png", dpi=200)
    pl.close(tfig)

    cfig, caxes = plot_corner(patchname)
    cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
    cfig.savefig(f"{dirname}/{tag}_corner.png", dpi=200)
    pl.close(cfig)

    rfig, raxes, rcb, val = plot_residual(patchname)
    rfig.savefig(f"{dirname}/{tag}_residual.png", dpi=200)
    pl.close(rfig)

    sys.exit()