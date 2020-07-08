#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import glob, sys, os
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
import h5py
from utils import make_chaincat


def get_results(fn):
    with h5py.File(fn, "r") as res:
        chain = res["chain"][:]
        #bands = res["bandlist"][:].astype("U").tolist()
        bands = ["Fclear"]
        ref = res["reference_coordinates"][:]
        active = res["active"][:]

    cat = make_chaincat(chain, bands, active, ref)
    return cat, active


def stats(cat, col):
    val = cat[col].mean(axis=-1) #/ cat["rhalf"].std(axis=-1)
    unc = cat[col].std(axis=-1)
    return val, unc


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--colname", type=str, default="rhalf")
    parser.add_argument("--patch_dir", type=str, default="../output/run3")
    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--min_snr", type=float, default=0)
    args = parser.parse_args()

    pl.ion()
    original = fits.getdata("../data/galsim_galaxy_grid_cat.fits")
    #res["lower_bound"][::7] / res["upper_bound"][::7]

    files = glob.glob(os.path.join(args.patch_dir, "patch????_results.h5"))
    res = [get_results(fn) for fn in files]
    cat = np.concatenate([r[0] for r in res])
    active = np.concatenate([r[1] for r in res])
    snr = original[active["source_index"]]["snr"]

    #good = ((np.abs(cat["ra"][:, -1] - original[active["source_index"]]["ra"]) < 5e-5) &
    #        (np.abs(cat["dec"][:, -1] - original[active["source_index"]]["dec"]) < 5e-5))
    good = snr >= args.min_snr
    cat = cat[good]
    active = active[good]
    orig = original[active["source_index"]]
    orig["q"] = np.sqrt(orig["q"])


    col = args.colname
    val, unc = stats(cat, col)
    delta = val - orig[col]
    lims = val.min(), val.max()

    ll = np.linspace(lims[0], lims[1], 10)
    snc = ["tomato", "orange", "royalblue", "green"]


    fig, ax = pl.subplots()
    ax.hist(delta / unc, bins=20, density=True, alpha=0.5, label=col)
    tag = r"$\sigma(\Delta_{{{c}}}/\sigma_{{{c}}})={s:4.2f}$"
    ax.text(0.1, 0.9, tag.format(c=col, s=np.std(delta / unc)),
            transform=ax.transAxes)
    ax.set_xlabel(r"$\Delta/\sigma$")
    ax.legend()


    fig, ax = pl.subplots()
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(snc)
    if "F" in col:
        rr = orig[col] * 0.02 * np.random.uniform(-1, 1, size=good.sum())
        ax.set_xlim(10, 5100)
        ax.set_ylim(10, 5100)
        ax.set_xscale("log")
        ax.set_yscale("log")
        xx = (orig[col] + rr)
        widths = 0.05 * xx
    else:
        d = np.diff(lims)
        rr = 0.03 * np.random.uniform(-d, d, size=good.sum())
        xx = (orig[col] + rr)
        widths = np.ones_like(xx) * 0.01 * d


    if args.violin:
        for i, snr in enumerate(np.sort(np.unique(orig["snr"]))):
            sel = orig["snr"] == snr
            x = xx[sel]
            y = cat[col][sel]
            vparts = ax.violinplot(y.T, positions=x, showextrema=False, widths=widths[sel],
                           showmedians=False, showmeans=False,)
            [p.set_facecolor(snc[i]) for p in vparts["bodies"]]
            [p.set_alpha(0.3) for p in vparts["bodies"]]
    else:
        ax.errorbar(orig[col] + rr, val, yerr=unc, linestyle="", color="k", alpha=0.5)
        cb = ax.scatter(orig[col] + rr, val, c=np.log10(orig["snr"]), cmap=cm, zorder=2)
        pl.colorbar(cb, label="log(S/N)")

    ax.plot(ll, ll, "k:")
    ax.set_xlabel(col + " (GalSim)")
    ax.set_ylabel(col + " (fpho)")