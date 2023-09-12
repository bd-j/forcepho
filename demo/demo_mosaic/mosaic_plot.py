#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

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


if __name__ == "__main__":

    show = ["CLEAR", "rhalf", "sersic"]

    sd = Samples("output/dither/dither_samples.h5")
    sm = Samples("output/mosaic/mosaic_samples.h5")

    shape = len(show)*len(sd.active), -1
    xxd = np.array([sd.chaincat[c] for c in show]).reshape(shape)
    xxm = np.array([sm.chaincat[c] for c in show]).reshape(shape)
    truths = np.atleast_2d(xxd[:, 0]).T

    labels = [f"{c}_{n}" for c in show for n in range(len(sd.active))]
    span = multispan([xxd[:, sd.n_tune:], xxm[:, sm.n_tune:]])

    kwargs = dict(hist_kwargs=dict(alpha=0.6, histtype="stepfilled"))

    pl.ion()
    fig, axes = pl.subplots(len(labels), len(labels), figsize=(12, 12))
    axes = corner(xxd[:, sd.n_tune:], axes, span=span, color="royalblue", **kwargs)
    axes = corner(xxm[:, sm.n_tune:], axes, span=span, color="darkorange", **kwargs)
    scatter(truths, axes, zorder=10, color="k", edgecolor="k")
    prettify_axes(axes, labels, label_kwargs=dict(fontsize=12), tick_kwargs=dict(labelsize=10))
    [ax.axvline(t, linestyle=":", color="k") for ax, t in zip(np.diag(axes), truths[:,0])]
    #mfig, maxes = pl.subplots(3, 3)
    #allcorner(xxm, labels=["CLEAR", "rhalf", "sersic"], axes=maxes)

    from matplotlib.patches import Patch
    legends = ["Dithers", "Mosaic"]
    artists = [Patch(color="royalblue", alpha=0.6), Patch(color="darkorange", alpha=0.6)]
    fig.legend(artists, legends, loc='upper right', bbox_to_anchor=(0.8, 0.8),
               frameon=True, fontsize=14)
    fig.savefig("mosaic_corner.png", dpi=300)