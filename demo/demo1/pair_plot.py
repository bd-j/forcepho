#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from forcepho.postprocess import Samples, Residuals
from prospect.plotting.corner import allcorner, scatter, marginal


def plot_corner(patchname, band="CLEAR", smooth=0.05, hkwargs=dict(alpha=0.65),
                dkwargs=dict(color="red", marker="."), axes=None):

    samples = Samples(patchname)
    xx = samples.chaincat["CLEAR"]
    labels = ["Source 1 Flux", "Source 2 Flux"]
    truth = np.atleast_2d(xx[:, 0])

    axes = allcorner(xx, labels, axes,
                     color="royalblue",  # qcolor="black",
                     psamples=truth.T,
                     smooth=smooth, hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)
    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")
        if (labels[i] == "ra") | (labels[i] == "dec"):
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            axes[-1, i].xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    #ymap = np.atleast_2d(samples.get_map(structured=True)["CLEAR"])
    #scatter(ymap.T, axes, zorder=20, color="k", marker=".")
    #for ax, val in zip(np.diag(axes), ymap[0]):
    #    ax.axvline(val, linestyle=":", color="k")
    #scatter(xx, axes, zorder=20, color="cyan", alpha=0.5, marker=".")

    return axes


def plot_residual(patchname, vmin=-3, vmax=10, rfig=None, raxes=None):
    s = Samples(patchname)
    r = Residuals(patchname.replace("samples", "residuals"))
    data, _, _ = r.make_exp(value="data")
    delta, _, _ = r.make_exp(value="residual")
    ierr, _, _ = r.make_exp(value="ierr")
    if raxes is None:
        rfig, raxes = pl.subplots(1, 3, sharex=True, sharey=True)
    kw = dict(origin="lower", vmin=vmin, vmax=vmax)
    cb = raxes[0].imshow((data * ierr).T, **kw)
    cb = raxes[1].imshow((delta * ierr).T, **kw)
    cb = raxes[2].imshow(((data-delta) * ierr).T, **kw)
    #rfig.colorbar(cb, label=r"$\chi=$ (Data - Model) / Unc")

    val = s.get_sample_cat(-1)
    return rfig, raxes, cb, val


def plot_both(patchname, show_current=True):

    fig = pl.figure(figsize=(6.5, 8.0))
    gs0 = GridSpec(48, 3, figure=fig)
    c = 18

    if True:
        gs_resid = gs0
        raxes = [fig.add_subplot(gs_resid[:c, 0])]
        raxes += [fig.add_subplot(gs_resid[:c, i], sharex=raxes[0], sharey=raxes[0])
                  for i in range(1, 3)]
        titles = ["Data", "Residual", "Model"]
        _, raxes, cb, val = plot_residual(patchname, raxes=raxes)
        for i in range(3):
            cax = fig.add_subplot(gs_resid[c+1:c+2, i])
            pl.colorbar(cb, cax=cax, orientation="horizontal", label=r"flux/$\sigma$")
            raxes[i].set_title(titles[i])

    if True:
        gs_corner = gs0
        d0, dd = c+8, int((48-(c+8)) / 2)
        paxes = np.array([fig.add_subplot(gs_corner[d0+i*dd:d0+(i+1)*dd, j])
                         for i in range(2) for j in range(2)]).reshape(2, 2)
        paxes = plot_corner(patchname, axes=paxes)

        if show_current:
            ymap = np.atleast_2d(val["CLEAR"])
            scatter(ymap.T, paxes, zorder=20, color="grey", marker=".")
            for i, ax in enumerate(np.diag(paxes)):
                ax.axvline(ymap[0, i], color="grey", alpha=0.5)

    return fig, raxes, paxes


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--patchname", type=str, default="")
    args = parser.parse_args()

    fig, raxes, paxes = plot_both(args.patchname)