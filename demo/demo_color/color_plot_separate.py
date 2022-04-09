#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, argparse
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from forcepho.postprocess import Samples, Residuals
from forcepho.utils import frac_sersic
from prospect.plotting.corner import allcorner


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


def color_corner(sample_list, bands=["BLUE", "RED"], smooth=0.05, hkwargs=dict(alpha=0.65),
                 dkwargs=dict(color="red", marker="."), axes=None):
    flux = {}
    for samples in sample_list:
        flux[samples.bands[0]] = apflux(samples)
        n_source = len(samples.active)
    color = -2.5 * np.log10(flux[bands[0]] / flux[bands[1]])
    xx = color
    labels = [f"[{bands[0]} - {bands[1]}]_{i+1}" for i in range(n_source)]
    truth = np.atleast_2d(xx[:, 0])

    axes = allcorner(xx[:, samples.n_tune:], labels, axes,
                     upper=True,
                     color="royalblue",
                     psamples=truth.T,
                     smooth=smooth,
                     hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)
    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")

    return axes


def apflux(samples, band=None, nradii=1.0):
    if band is None:
        band = samples.bands[0]
    x = frac_sersic((samples.active["rhalf"] * nradii)[:, None],
                    sersic=samples.chaincat["sersic"],
                    rhalf=samples.chaincat["rhalf"])
    return np.squeeze(samples.chaincat[band] * x)


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


def plot_both(patchnames, band=["BLUE", "RED"], show_current=True):

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
        for j, p in enumerate(patchnames):
            _, _, cb, val = plot_residual(p, raxes=raxes[j:j+1, :])

        titles = ["Data", "Residual", "Model"]
        for i, rax in enumerate(raxes[0]):
            cax = fig.add_subplot(gs_resid[c:c+1, i])
            pl.colorbar(cb, cax=cax, orientation="horizontal", label=r"flux/$\sigma$")
            rax.set_title(titles[i])
        for j, rax in enumerate(raxes[:, 0]):
            rax.text(0.5, 0.9, band[j], color="magenta", transform=rax.transAxes)

    if True:
        samples = [Samples(p) for p in patchnames]

        nx, ny = 4, 3
        gs_corner = GridSpecFromSubplotSpec(ny, nx, subplot_spec=gs0[1])
        paxes = np.array([fig.add_subplot(gs_corner[i, j])
                         for i in range(ny) for j in range(nx)]).reshape(ny, nx)

        taxes = paxes[-2:, :2]
        #taxes = total_corner(samples, axes=taxes)
        caxes = paxes[:2, -2:]
        caxes = color_corner(samples, axes=caxes)

        empty = paxes[0, :2].tolist() + paxes[-1, -2:].tolist() + taxes.flatten().tolist()
        [ax.set_frame_on(False) for ax in empty]
        [ax.set_xticks([]) for ax in empty]
        [ax.set_yticks([]) for ax in empty]

    return fig, raxes, paxes


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--joint_patch", type=str, default="output/together_v1/patches/patch1_samples.h5")
    parser.add_argument("--red_patch", type=str, default="output/separate_v1/patches/patch_RED_samples.h5")
    parser.add_argument("--blue_patch", type=str, default="output/separate_v1/patches/patch_BLUE_samples.h5")
    args = parser.parse_args()

    patchnames = [args.blue_patch, args.red_patch]
    fig, raxes, paxes = plot_both(patchnames)

