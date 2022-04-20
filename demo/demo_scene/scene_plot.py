#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
# ell = Ellipse(xy=mu, width=w, height=h, angle=theta)

from astropy.io import fits
from forcepho.superscene import LinkedSuperScene, rectify_catalog, sourcecat_dtype, convert_pa


def convert_3dhst(incat, outcat):
    ra, dec, width = 53.13, -27.84, 0.01
    bands = ["F435W", "F606W", "F775W", "F850LP", "F125W", "F140W", "F160W"]
    catalog = fits.getdata(incat)
    sel = ((catalog["ra"] > (ra - width)) &
           (catalog["ra"] < (ra + width)) &
           (catalog["dec"] > (dec - width)) &
           (catalog["dec"] < (dec + width)))
    catalog = catalog[sel]
    dtype = sourcecat_dtype(bands=bands)
    cat = np.zeros(len(catalog), dtype=dtype)
    cat["ra"] = catalog["ra"]
    cat["dec"] = catalog["dec"]
    cat["q"] = np.sqrt(catalog["b_image"] / catalog["a_image"])
    cat["pa"] = convert_pa(catalog["theta_j2000"], from_deg=True, reverse=True)  # these are already N of W
    cat["sersic"] = 2.0  # placeholder
    cat["rhalf"] = catalog["flux_radius"] * 0.06  # from pix to arcsec
    # TODO: use an isophote for this based on rhalf, flux/2, and n.
    cat["roi"] = catalog["kron_radius"] * 3 * 0.06
    for b in bands:
        # TODO: don't clip but use unc to set bounds
        cat[b] = np.clip(catalog[f"f_{b.lower()}"], 1e-5, np.inf)

    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(cat)])
    for hdu in hdul:
        hdu.header["FILTERS"] = ",".join(bands)
    hdul.writeto(outcat, overwrite=True)
    hdul.close()


def show_reg(reg, ax, color, text=""):

    ell = Ellipse(xy=(reg.ra, reg.dec), height=reg.radius*2, width=reg.radius*2)
    ell.set_edgecolor(color)
    ell.set_facecolor("none")
    ell.set_linewidth(1.0)
    ax.add_artist(ell)
    if text:
        ax.annotate(text, (reg.ra+reg.radius, reg.dec+reg.radius), color=color)


if __name__ == "__main__":

    # --- read and convert to forcepho format ---
    incat = "./goodss_3dhst.v4.1.cat.FITS"
    outcat = "goodss-3dhst-example.fits"
    if not os.path.exists(outcat):
        convert_3dhst(incat, outcat)

    cat, bands, hdr = rectify_catalog(outcat)
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               maxactive_per_patch=15, strict=True,
                               minradius=1., maxradius=6., buffer=0.5)

    kw = dict(marker=".", linestyle="")
    fig, axes = pl.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    # show superscene (with rois in gray)
    ax = axes[0]
    ax.plot(sceneDB.sourcecat["ra"], sceneDB.sourcecat["dec"], color="grey", **kw)

    # show a superscene with two checked out scenes (active/fixed)
    ax = axes[1]
    ax.plot(sceneDB.sourcecat["ra"], sceneDB.sourcecat["dec"], color="grey", **kw)
    patches = []
    for seed in range(2):
        region, active, fixed = sceneDB.checkout_region()
        patches.append((region, active, fixed))

    for i, p in enumerate(patches):
        reg, active, fixed = p
        ax.plot(active["ra"], active["dec"], color="firebrick", **kw)
        ax.plot(fixed["ra"], fixed["dec"], color="darkslateblue", **kw)
        show_reg(reg, ax, "firebrick", i+1)

    # check in one scene and check out a new one
    ax = axes[2]
    ax.plot(sceneDB.sourcecat["ra"], sceneDB.sourcecat["dec"], color="grey", **kw)

    reg, active, fixed = patches.pop(0)
    sceneDB.checkin_region(active, fixed=fixed, niter=200)
    ax.plot(active["ra"], active["dec"], color="darkorange", **kw)

    region, active, fixed = sceneDB.checkout_region()
    patches.append((region, active, fixed))

    for i, p in enumerate(patches):
        reg, active, fixed = p
        ax.plot(active["ra"], active["dec"], color="firebrick", **kw)
        ax.plot(fixed["ra"], fixed["dec"], color="darkslateblue", **kw)
        show_reg(reg, ax, "firebrick", i+2)

    [ax.set_xlabel(r"$\alpha$ (deg.)") for ax in axes]
    axes[0].set_ylabel(r"$\delta$ (deg.)")

    art = [Line2D([], [], color="gray", **kw), Line2D([], [], color="firebrick", **kw),
           Line2D([], [], color="darkslateblue", **kw), Line2D([], [], color="darkorange", **kw)]
    labels = ["Available", "Active", "Fixed", "Sampled"]

    axes[2].legend(art, labels, loc="lower left",
                   frameon=True, framealpha=1.0)

    fig.savefig("scenes.png", dpi=400)