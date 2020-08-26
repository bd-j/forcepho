#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits


def legend(row):
    fmt = ("flux={:.1f}\n"
           "x,y={:.0f},{:.0f}\n"
           "sersic={:.1f}\n"
           "rhalf={:.2f}\n"
           "b/a={:.2f}"
           )
    s = fmt.format(row["Fclear"], row["x"], row["y"],
                   row["sersic"], row["rhalf"],
                   row["q"])
    return s


def show_one(row, truth, model, axes=None, n_pix_per_gal=24, fontsize=8,
             **imshow_kwargs):
    d = np.array([-1, 1])

    xs, xf = (row["x"] + d*n_pix_per_gal).astype(int)
    ys, yf = (row["y"] + d*n_pix_per_gal).astype(int)
    t = truth[ys:yf, xs:xf]
    m = model[ys:yf, xs:xf]
    r = t - m

    # --- show the image ---
    ax = axes[0]
    cm = ax.imshow(r, origin="lower", **imshow_kwargs)

    ax.text(0.05, 0.95, legend(row), transform=ax.transAxes,
            verticalalignment="top", fontsize=fontsize)
    r = "$\Sigma\Delta={:0.1f}$\n$\Sigma\Delta^2={:.1f}$\nN={:.0f}"
    s = r.format(im.sum(), (im**2).sum(), np.prod(im.shape))
    ax.text(0.6, 0.95, s, transform=ax.transAxes, verticalalignment="top",
            fontsize=fontsize)

    # --- show slices through the center ---
    if len(axes) > 1:
        ax = axes[1]
        ax.plot(t[n_pix_per_gal, :], label="GalSim")
        ax.plot(m[n_pix_per_gal, :], label="fpho")
        ax.legend()
        ax = axes[2]
        ax.plot(t[:, n_pix_per_gal], label="GalSim")
        ax.plot(m[:, n_pix_per_gal], label="fpho")
        ax.legend()

    return ax, cm


if __name__ == "__main__":

    pl.ion()

    cat = fits.getdata("../data/galsim_galaxy_grid_cat.fits")
    truth = fits.getdata("../data/galsim_galaxy_grid_truth.fits")
    noisy = fits.getdata("../data/galsim_galaxy_grid_noisy.fits")
    model = fits.getdata("../output/galsim_galaxy_grid_force.fits")
    unc = fits.getheader("../data/galsim_galaxy_grid_noisy.fits")["NOISE"]
    chi = (noisy - model) / unc

    ns = np.unique(cat["sersic"])

    #rcParams = plot_defaults(rcParams)
    nrow, ncol = 6, 4
    figsize = (9, 13)  # 14, 8.5
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, ncol, height_ratios=nrow * [10],
                  hspace=0.05, wspace=0.02,
                  left=0.08, right=0.95, bottom=0.13, top=0.95)
    gsc = GridSpec(1, 1, left=0.08, right=0.95, hspace=0.2,
                   bottom=0.07, top=0.1)
                   #bottom=0.89, top=0.95)
    axes = np.array([fig.add_subplot(gs[i, j])
                     for i in range(nrow) for j in range(ncol)]).reshape(nrow, ncol)

    residual = truth-model
    for j, n in enumerate(ns):
        sel = (cat["snr"] > 31) & (cat["sersic"] == n)
        for i, row in enumerate(cat[sel]):
            ax = axes[i, j]
            ax, cm = show_one(row, residual, axes=[ax], vmin=-5, vmax=5)
            #ax, cm = show_one(row, truth, ax=ax, vmin=0, vmax=50)

    cax = fig.add_subplot(gsc[0, 0])
    fig.colorbar(cm, cax=cax, label="GalSim-fpho", orientation="horizontal")
    [ax.set_xticklabels("") for ax in axes[:-1, :].flat]
    [ax.set_yticklabels("") for ax in axes[:, 1:].flat]

    fig.suptitle("Pixel Scale = 0.03")
    sys.exit()
    fig.savefig("galsim_fpho_residuals.pdf")