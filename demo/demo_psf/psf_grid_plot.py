#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob, argparse
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from forcepho.postprocess import Samples


__all__ = ["get_map", "aperture_flux", "make_catalog",
           "compare_parameters", "compare_apflux"]

fsize = 8, 9.5


def get_map(s):
    lnp = s.stats["model_logp"]
    ind_ml = np.argmax(lnp)
    #row_map = s.get_sample_cat(ind_ml)[0]
    #ymap = np.atleast_2d([row_map[c] for c in s.bands + s.shape_cols])
    ymap = np.atleast_2d(s.chain[ind_ml, :])

    return ymap


def make_catalog(tagnames, n_full=0, bands=["CLEAR"]):

    # Get catalog data type
    for tag in tagnames:
        if os.path.exists(f"{tag}_samples.h5"):
            s = Samples(f"{tag}_samples.h5")
            break
        else:
            continue
    n_sample, shapes = s.n_sample, s.shape_cols
    scols = bands + shapes
    icols = [("id", "<i4"), ("wall", "<f4"), ("lnp", "<f8", n_sample)]

    n_out = len(tagnames)
    new = np.dtype(icols + [(c, float, n_sample) for c in scols])
    cat = np.zeros(n_out, new)

    # Make and fill the catalog
    cat["id"] = -1
    for p, tag in enumerate(tagnames):
        if not os.path.exists(f"{tag}_samples.h5"):
            continue
        s = Samples(f"{tag}_samples.h5")
        if s.n_sample != n_sample:
            print(f"sizes do not match for patch {p}")
            continue
        cat["id"][p] = p
        cat["wall"][p] = s.wall_time
        cat["lnp"][p] = s.stats["model_logp"][-n_sample:]
        for col in shapes + s.bands:
            cat[col][p] = s.chaincat[col][:, -n_sample:]

    return cat


def compare_parameters(scat, tcat, parname, add_jitter=True,
                       point_type="median", colorby="fwhm", splitby="snr",
                       as_delta=False, dfax=None, onlymax=True):

    colors = np.unique(tcat[colorby])
    if splitby:
        splits = np.unique(tcat[splitby])
    else:
        splits = [slice(None)]

    # HACK
    if onlymax:
        splits = [splits[-1]]

    # --- xcoordinate ---
    xtrue = tcat[parname].copy()
    #xr = x.min()*0.9, x.max()*1.1
    xr = xtrue.min(), xtrue.max()
    dx = np.diff(xr)
    if add_jitter:
        jitter = np.random.uniform(-dx*0.02, dx*0.02, len(xtrue))
        x = xtrue + jitter
        xr = x.min(), x.max()
    else:
        x = xtrue
    linex = np.linspace(*xr)

    # --- y-coordinate ---
    yy = scat[parname]
    y = np.percentile(yy, [16, 50, 84], axis=-1)
    if point_type == "map":
        ind_ml = np.argmax(scat["lnp"], axis=-1)
        ymap = yy[np.arange(len(yy)), ind_ml]
        y[1, :] = ymap
    if as_delta:
        y = y - xtrue
        liney = np.zeros_like(linex)
    else:
        liney = linex

    # --- Fig and axes ---
    if dfax is None:
        dfig, daxes = pl.subplots(len(splits), 1, sharex=True,
                                  figsize=(8, 1+3*len(splits)))
    else:
        dfig, daxes = dfax
    daxes = np.atleast_1d(daxes)

    # --- loop over splits ---
    for i, s in enumerate(splits):
        ax = daxes[i]
        sel = (scat["id"] >= 0) & (tcat[splitby] == s)
        ax.errorbar(x[sel], y[1, sel], np.diff(y, axis=0)[:, sel],
                    marker="", linestyle="", color="gray", alpha=0.7, zorder=0)
        cb = ax.scatter(x[sel], y[1, sel], c=tcat[colorby][sel], alpha=0.75,
                        vmin=colors.min(), vmax=colors.max())
        ax.plot(linex, liney, "k:")
        if splitby is not None:
            ax.text(0.8, 0.2, f"{splitby.upper()}={s}", transform=ax.transAxes)

    #dfig.colorbar(cb, label=colorby, orientation="vertical", ax=daxes)
    if as_delta:
        [ax.set_ylabel(f"$\Delta${parname} (output-input)") for ax in daxes.flat]
    else:
        [ax.set_ylabel(f"{parname} (output)") for ax in daxes.flat]
    daxes.flat[-1].set_xlabel(f"{parname} (input)")

    return dfig, daxes, cb


def compare_apflux(scat, tcat, band=["CLEAR"], colorby="fwhm"):
    aflux, tflux = aperture_flux(scat, tcat, band=band)
    aflux = aflux[0]
    ffig, faxes = pl.subplots(figsize=(8, 4))
    print(aflux.shape)
    print(aflux.mean(axis=-1).shape, tcat[colorby].shape)

    jitter = np.random.uniform(0.9, 1.1, len(tcat))
    x = tcat["snr"] * jitter
    yy = aflux * 2.0
    y = np.percentile(yy, [16, 50, 84], axis=-1)
    print(x.shape, y.shape)

    faxes.errorbar(x, y[1, :], np.diff(y, axis=0),
                   marker="", linestyle="", color="gray", alpha=0.7, zorder=0)
    cb = faxes.scatter(x, yy.mean(axis=-1),
                       c=tcat[colorby], alpha=0.8)

    ffig.colorbar(cb, orientation="vertical", label=colorby)
    faxes.axhline(1.0, color="k", linestyle=":")
    faxes.set_xlabel("SNR")
    faxes.set_ylabel("forcepho aperture flux (50th pctile) / true aperture flux")
    faxes.set_xscale("log")
    ffig.tight_layout()
    return ffig, faxes


def aperture_flux(scat, truths, band=["CLEAR"]):
    from forcepho.utils import frac_sersic
    rhalf = truths["rhalf"]
    fr = frac_sersic(rhalf[:, None], sersic=scat["sersic"], rhalf=scat["rhalf"])
    total_flux = np.array([scat[b] for i, b in enumerate(band)])
    aperture_flux = total_flux * fr

    return aperture_flux, total_flux


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_name", type=str, default="./grid.fits")
    parser.add_argument
    config = parser.parse_args()

    # Make summary plots
    tcat = fits.getdata(config.grid_name)
    tags = make_all_tags(tcat, config)
    tags = [os.path.join(config.banddir, tag, tag) for tag in tags]
    scat = make_catalog(tags, bands=config.bands)
    fits.writeto(os.path.join(config.banddir, "ensemble_chains.fits"), scat)

    comp = [("rhalf", "sersic"), ("sersic", "rhalf"), ("q", "rhalf")]
    for show, by in comp:
        fig, axes = pl.subplots(2, 1, sharex=True)
        fig, ax, cb = compare_parameters(scat, tcat, show, dfax=(fig, axes[0]),
                                        splitby="snr", colorby=by)
        fig, ax, cb = compare_parameters(scat, tcat, show, dfax=(fig, axes[1]),
                                        as_delta=True, splitby="snr", colorby=by)
        fig.colorbar(cb, label=by, ax=axes, orientation="vertical")
        fig.suptitle(config.bandname)
        fig.savefig(os.path.join(config.banddir, f"{config.bandname.lower()}_{show}_comparison.pdf"))
        pl.close(fig)

    fig, axes = compare_apflux(scat, tcat, band=config.bands, colorby="rhalf")
    fig.suptitle(config.bandname)
    fig.savefig(os.path.join(config.banddir, f"{config.bandname.lower()}_flux_comparison.pdf"), dpi=200)