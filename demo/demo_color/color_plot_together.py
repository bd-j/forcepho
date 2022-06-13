#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap, colorConverter

from forcepho.postprocess import Samples, Residuals
from prospect.plotting.corner import allcorner


def total_corner(samples, bands=["BLUE", "RED"], smooth=0.05, hkwargs=dict(alpha=0.65),
                 dkwargs=dict(color="red", marker="."), axes=None):

    n_source = len(samples.active)
    total = np.array([samples.chaincat[b].sum(axis=0) for b in bands])
    xx = total
    labels = [f"{b}_total" for b in bands]
    truth = np.atleast_2d(xx[:, 0])

    axes = allcorner(xx[:, samples.n_tune:], labels, axes,
                     #upper=False,
                     color="royalblue",
                     psamples=truth.T,
                     smooth=smooth,
                     hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)

    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")


def color_corner(samples, bands=["BLUE", "RED"], smooth=0.05, hkwargs=dict(alpha=0.65),
                 dkwargs=dict(color="red", marker="."), axes=None):
    n_source = len(samples.active)
    color = -2.5 * np.log10(samples.chaincat[bands[0]] / samples.chaincat[bands[1]])
    xx = color
    labels = [f"[{bands[0]} - {bands[1]}]_{i+1}" for i in range(n_source)]
    truth = np.atleast_2d(xx[:, 0])

    axes = allcorner(xx[:, samples.n_tune:], labels, axes,
                     #upper=True,
                     color="royalblue",
                     psamples=truth.T,
                     smooth=smooth,
                     hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)

    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")

    return axes


def plot_residual(patchname, vmin=-3, vmax=10, rfig=None, raxes=None):
    s = Samples(patchname)
    r = Residuals(patchname.replace("samples", "residuals"))

    if raxes is None:
        rfig, raxes = pl.subplots(nexp, 3, sharex=True, sharey=True)

    for i, e in enumerate(r.exposures):
        data, _, _ = r.make_exp(i, value="data")
        delta, _, _ = r.make_exp(i, value="residual")
        ierr, _, _ = r.make_exp(i, value="ierr")
        kw = dict(origin="lower", vmin=vmin, vmax=vmax)
        cb = raxes[i, 0].imshow((data * ierr).T, **kw)
        cb = raxes[i, 1].imshow((delta * ierr).T, **kw)
        cb = raxes[i, 2].imshow(((data-delta) * ierr).T, **kw)

    val = s.get_sample_cat(-1)
    return rfig, raxes, cb, val


def plot_both(patchname, band=["BLUE", "RED"], show_current=True):

    fig = pl.figure(figsize=(8, 13.5))
    gs0 = GridSpec(2, 1, figure=fig)

    nexp = 2

    if True:
        r = 20
        c = nexp * r
        gs_resid = GridSpecFromSubplotSpec(c+1, 3, subplot_spec=gs0[0], hspace=1.0)
        raxes = []
        for j in range(nexp):
            raxes += [fig.add_subplot(gs_resid[r*j:r*(j+1), 0])]
            raxes += [fig.add_subplot(gs_resid[r*j:r*(j+1), i], sharex=raxes[-1], sharey=raxes[-1])
                      for i in range(1, 3)]
        raxes = np.array(raxes).reshape(nexp, 3)
        titles = ["Data", "Residual", "Model"]
        _, raxes, cb, val = plot_residual(patchname, raxes=raxes)
        for i, rax in enumerate(raxes[0]):
            cax = fig.add_subplot(gs_resid[c:c+1, i])
            pl.colorbar(cb, cax=cax, orientation="horizontal", label=r"flux/$\sigma$")
            rax.set_title(titles[i])
        for j, rax in enumerate(raxes[:, 0]):
            rax.text(0.5, 0.9, band[j], color="magenta", transform=rax.transAxes)

    if True:
        samples = Samples(patchname)
        nx, ny = 4, 3
        gs_corner = GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[1])
        paxes = np.array([fig.add_subplot(gs_corner[i, j])
                         for i in range(ny) for j in range(nx)]).reshape(ny, nx)

        taxes = paxes[-2:, :2]
        taxes = total_corner(samples, axes=taxes)
        caxes = paxes[:2, -2:]
        caxes = color_corner(samples, axes=caxes)

        empty = paxes[0, :2].tolist() + paxes[-1, -2:].tolist()
        [ax.set_frame_on(False) for ax in empty]
        [ax.set_xticks([]) for ax in empty]
        [ax.set_yticks([]) for ax in empty]

    return fig, raxes, paxes


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--patchname", type=str, default="output/together_v1/patches/patch_BLUE+RED_samples.h5")
    args = parser.parse_args()

    fig, raxes, paxes = plot_both(args.patchname)
    fig.savefig(args.patchname.split("/")[-3] + ".png")