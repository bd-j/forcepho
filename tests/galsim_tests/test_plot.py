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


def make_catalog(tagnames, n_full=0, bands=["CLEAR"]):

    # Get catalog data type
    s = Samples(f"{tagnames[0]}_samples.h5")
    n_sample, shapes = s.n_sample, s.shape_cols
    scols = bands + shapes
    icols = [("id", "<i4"), ("wall", "<f4"), ("lnp", "<f8", n_sample)]

    n_out = len(tagnames)
    new = np.dtype(icols + [(c, float, n_sample) for c in scols])
    cat = np.zeros(n_out, new)

    # Make and fill the catalog
    cat["id"] = -1
    for p, tag in enumerate(tagnames):
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


def compare_parameters(scat, tcat, parname, point_type="median",
                       colorby="fwhm", splitby="snr"):

    splits = np.unique(tcat[splitby])
    colors = np.unique(tcat[colorby])

    # xcoordinate
    x = tcat[parname].copy()
    xr = x.min()*0.9, x.max()*1.1
    dx = np.diff(xr)
    jitter = np.random.uniform(-dx*0.02, dx*0.02, len(x))
    x = x + jitter
    line = np.linspace(*xr)

    yy = scat[parname]
    y = np.percentile(yy, [16, 50, 84], axis=-1)
    if point_type == "map":
        ind_ml = np.argmax(scat["lnp"], axis=-1)
        ymap = yy[np.arange(len(yy)), ind_ml]
        y[1, :] = ymap

    dfig, daxes = pl.subplots(len(splits), 1, sharex=True,
                              figsize=(8, 1+3*len(splits)))
    for i, s in enumerate(splits):
        ax = daxes[i]
        sel = (scat["id"] >= 0) & (tcat[splitby] == s)
        ax.errorbar(x[sel], y[1, sel], np.diff(y, axis=0)[:, sel],
                    marker="", linestyle="", color="gray")
        cb = ax.scatter(x[sel], y[1, sel], c=tcat[colorby][sel], alpha=0.5,
                        vmin=colors.min(), vmax=colors.max())
        ax.plot(line, line, "k:")
        ax.text(0.8, 0.2, f"{splitby.upper()}={s}", transform=ax.transAxes)

    dfig.colorbar(cb, label=colorby, orientation="vertical", ax=daxes)
    [ax.set_ylabel(f"{parname} (forcepho)") for ax in daxes.flat]
    daxes.flat[-1].set_xlabel(f"{parname} (input)")

    return dfig, daxes


def compare_apflux(scat, tcat, colorby="fwhm"):
    aflux, tflux = aperture_flux(scat, tcat)
    ffig, faxes = pl.subplots(figsize=(8, 4))
    print(aflux.mean(axis=-1).shape, tcat[colorby].shape)
    cb = faxes.scatter(tcat["snr"] * np.random.uniform(0.9, 1.1, len(tcat)),
                       aflux.mean(axis=-1) * 2.0, c=tcat[colorby], alpha=0.5)
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
    total_flux = np.array([scat[i][b] for i, b in enumerate(band)])
    aperture_flux = total_flux * fr

    return aperture_flux, total_flux


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