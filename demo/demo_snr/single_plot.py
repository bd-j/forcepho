#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from forcepho.postprocess import Samples, Residuals
from forcepho.utils.corner import allcorner, scatter, marginal, corner, get_spans, prettify_axes


def multispan(parsets):
    spans = []
    for x in parsets:
        spans.append(get_spans(None, x, weights=None))
    spans = np.array(spans)
    span = spans[:, :, 0].min(axis=0), spans[:, :, 1].max(axis=0)
    span = tuple(np.array(span).T)
    return span


def plot_corner(patchnames, config, band="CLEAR", smooth=0.05):

    legends = [f"S/N={s:.0f}" for s in config.snrlist]
    colors = ["slateblue", "darkorange", "firebrick", "grey", "cornflowerblue"]

    labels = ["Flux", r'R$_{half}$ (")', r"$n_{\rm sersic}$", r"$\sqrt{b/a}$", r"PA (radians)"]
    show = [band, "rhalf", "sersic", "q", "pa"]
    print(show)
    xx = []
    for name in patchnames:
        s = Samples(name)
        x = np.array([s.chaincat[c][0] for c in show])
        xx.append(x)
        n_tune = s.n_tune

    span = multispan([x[:, n_tune:] for x in xx])
    kwargs = dict(hist_kwargs=dict(alpha=0.65, histtype="stepfilled"))

    truths = np.atleast_2d(xx[0][:, 0]).T

    fig, axes = pl.subplots(len(labels), len(labels), figsize=(12, 12))
    for x, color in zip(xx, colors[:len(config.snrlist)]):
        axes = corner(x[:, n_tune:], axes, span=span, color=color, **kwargs)
    scatter(truths, axes, zorder=20, marker="o", color="k", edgecolor="k")
    prettify_axes(axes, labels, label_kwargs=dict(fontsize=12), tick_kwargs=dict(labelsize=10))
    [ax.axvline(t, linestyle=":", color="k") for ax, t in zip(np.diag(axes), truths[:, 0])]

    from matplotlib.patches import Patch
    artists = [Patch(color=color, alpha=0.6) for color in colors]
    fig.legend(artists, legends, loc='upper right', bbox_to_anchor=(0.8, 0.8),
               frameon=True, fontsize=14)

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


def plot_traces(patchname, fig=None, axes=None):
    s = Samples(patchname)
    if axes is None:
        fig, axes = pl.subplots(7, 1, sharex=True)
    truth = s.get_sample_cat(0)
    s.show_chain(axes=axes, truth=truth, bandlist=["CLEAR"])

    span = 0.999999426697
    q = 100 * np.array([0.5 - 0.5 * span, 0.5 + 0.5 * span])
    lim = np.percentile(s.chaincat["CLEAR"], list(q))
    axes[0].set_ylim(*lim)

    labels = [r"Flux", r"RA", r"Dec", r"$\sqrt{b/a}$", r"PA (radians)", r"$n_{\rm sersic}$", r'R$_{half}$ (")']
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])
        y = ax.get_ylim()
        ax.fill_betweenx(y, [0, 0], [s.n_tune, s.n_tune], alpha=0.3, color="gray")
    ax.set_xlim(0, s.chain.shape[0])
    ax.set_xlabel("HMC iteration")

    return fig, axes


if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--snrlist", type=float, nargs="*", default=[10, 30, 100])
    config = parser.parse_args()

    patchnames = [f"./output/v1/patches/patch_single_snr{s:03.0f}_samples.h5"
                  for s in config.snrlist]

    fig, axes = plot_corner(patchnames, config)
    fig.savefig("corner_snr.png", dpi=300)
    pl.close(fig)

    fig, axes, cb, val = plot_residual(patchnames[1])
    fig.suptitle("S/N="+'{:.0f}'.format(config.snrlist[1]))
    fig.savefig("residuals.png", dpi=300)
    pl.close(fig)

    fig, axes = pl.subplots(7, 1, sharex=True, figsize=(5, 8))
    fig, axes = plot_traces(patchnames[1], fig=fig, axes=axes)
    fig.suptitle("S/N="+'{:.0f}'.format(config.snrlist[1]))
    fig.savefig("trace.png", dpi=300)
    pl.close(fig)
