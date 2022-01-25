#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, os, sys
import argparse
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter

from astropy.io import fits
from forcepho.postprocess import Samples, Residuals, postsample_catalog

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

    # this doesn't do anything
    #[ax.set_xlabel(ax.get_xlabel(), labelpad=200) for ax in axes[-1,:]]
    #[ax.set_ylabel(ax.get_ylabel(), labelpad=30) for ax in axes[:, 0]]

    return fig, axes


def plot_residual(patchname):
    s = Samples(patchname)
    r = Residuals(patchname.replace("samples", "residuals"))
    delta, _, _ = r.make_exp(value="residual")
    ierr, _, _ = r.make_exp(value="ierr")
    rfig, rax = pl.subplots()
    cb = rax.imshow((delta * ierr).T, origin="lower")
    rfig.colorbar(cb, label=r"$\chi=$ (Data - Model) / Unc")

    val = s.get_sample_cat(-1)
    return rfig, rax, val


def get_map(s):
    lnp = s.stats["model_logp"]
    ind_ml = np.argmax(lnp)
    #row_map = s.get_sample_cat(ind_ml)[0]
    #ymap = np.atleast_2d([row_map[c] for c in s.bands + s.shape_cols])
    ymap = np.atleast_2d(s.chain[ind_ml, :])

    return ymap


def make_catalog(root, n_full=0, bands=["F090W", "F200W", "F277W", "F356W"]):
    avail = glob.glob(f"{root}/patches/*samples.h5")
    pids = np.sort([int(os.path.basename(a).split("_")[0][5:]) for a in avail])

    # Get catalog data type
    n_sample = Samples(f"{root}/patches/patch{pids[0]}_samples.h5").n_sample
    shapes = Samples(f"{root}/patches/patch{pids[0]}_samples.h5").shape_cols
    scols = bands + shapes
    icols = [("id", "<i4"), ("wall", "<f4"), ("lnp", "<f8", n_sample)]

    if n_full:
        n_out = n_full
    else:
        n_out = len(pids)
    new = np.dtype(icols + [(c, float, n_sample) for c in scols])
    cat = np.zeros(n_out, new)

    # Make and fill the catalog
    cat["id"] = -1
    for p in pids:
        s = Samples(f"{root}/patches/patch{p}_samples.h5")
        if s.n_sample != n_sample:
            print(f"sizes do not match for patch {p}")
            continue
        cat["id"][p] = p
        cat["wall"][p] = s.wall_time
        cat["lnp"][p] = s.stats["model_logp"][-n_sample:]
        for col in shapes + s.bands:
            cat[col][p] = s.chaincat[col][:, -n_sample:]

    return cat


def snr_plot(aflux, tflux, truths, bands=[], dx=0.05):

    aunc = aflux.std(axis=-1)
    tunc = tflux.std(axis=-1)
    x = 1 / truths["snr"]
    dx = np.diff([x.min(), x.max()])
    jitter = np.random.uniform(-dx*0.02, dx*0.02, len(x))
    sfig, saxes = pl.subplots(1, 2, figsize=(8, 4), sharey=True)
    for b in bands:
        sel = truths["band"] == b
        saxes[0].plot(tunc[sel], aunc[sel], marker="o", alpha=0.5, label=b, linestyle="")
        saxes[1].plot((x + jitter)[sel], aunc[sel], marker="o", alpha=0.5, label=b, linestyle="")

    xx = np.linspace(0.00, 0.1 * 5, 50)
    saxes[0].plot(xx, xx/5, ":k", label="y=x/5")
    saxes[0].legend()
    saxes[0].set_xlabel(r"$\sigma_{\rm samples}$ [Total Flux]")
    saxes[0].set_ylabel(r"$\sigma_{\rm samples}$ [Flux within true half-light radius]")

    saxes[1].plot(xx, xx, ":k", label="y=x")
    saxes[1].set_xlabel(r"1/SNR")
    saxes[1].legend()
    saxes[1].set_xlim(-0.02, 0.15)
    saxes[0].set_ylim(-0.02, 0.15)
    sfig.tight_layout()

    return sfig, saxes


def compare_parameters(scat, truths, parname, point_type="median",
                       colorby="band", splitby="snr"):

    splits = np.unique(truths[splitby])
    colors = np.unique(truths[colorby])

    yy = scat[parname]
    y = np.percentile(yy, [16, 50, 84], axis=-1)
    ind_ml = np.argmax(scat["lnp"], axis=-1)
    ymap = yy[np.arange(len(yy)), ind_ml]
    x = truths[parname].copy()
    xr = x.min()*0.9, x.max()*1.1
    dx = np.diff(xr)

    jitter = np.random.uniform(-dx*0.02, dx*0.02, len(x))
    x = x + jitter

    dfig, daxes = pl.subplots(len(splits), 1, sharex=True, figsize=fsize)
    for i, s in enumerate(splits):
        ax = daxes[i]
        for i, c in enumerate(colors):
            sel = (scat["id"] >= 0) & (truths[splitby] == s) & (truths[colorby] == c)
            if point_type == "median":
                ax.plot(x[sel], y[1, sel], marker="o", linestyle="", alpha=0.5, label=f"{colorby}={c}")
            else:
                ax.plot(x[sel], ymap[sel], marker="o", linestyle="", alpha=0.5, label=f"{colorby}={c}")
            ax.errorbar(x[sel], y[1, sel], np.diff(y, axis=0)[:, sel],
                        marker="", linestyle="", color="gray")
            line = np.linspace(*xr)
            ax.plot(line, line, "k:")
        ax.text(0.8, 0.2, f"{splitby.upper()}={s}", transform=ax.transAxes)
    dfig.suptitle(parname)
    daxes[-1].set_xlabel(f"{parname} (input)")
    [ax.set_ylabel(f"{parname} (forcepho)") for ax in daxes]

    return dfig, daxes


def aperture_flux(scat, truths):
    from forcepho.utils import frac_sersic
    band = truths["band"]
    rhalf = truths["rhalf"]
    fr = frac_sersic(rhalf[:, None], sersic=scat["sersic"], rhalf=scat["rhalf"])
    total_flux = np.array([scat[i][b] for i, b in enumerate(band)])
    aperture_flux = total_flux * fr

    return aperture_flux, total_flux


if __name__ == "__main__":

    # bad: 806
    # good: 836, 566
    # middling: 802

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="output/test_sampling_v1")
    parser.add_argument("--patch_index", type=int, default=-1)
    parser.add_argument("--parname", type=str, default="",
                        choices=["", "rhalf", "sersic", "q"])
    parser.add_argument("--point_type", type=str, default="median",
                        choices=["median", "map"])
    args = parser.parse_args()

    pl.ion()
    gridfile = glob.glob(f"{args.root}/*grid.fits")[0]
    truths = fits.getdata(gridfile)

    if args.patch_index > 0:

        #pids = np.arange(0, args.patch_index)
        pids = [args.patch_index]
        for pid in pids:
            title_fmt = "band={band}, SNR={snr},\nnsersic={sersic:.1f}, rhalf={rhalf:.2f}, q={q:.2f}"
            patchname = f"{args.root}/patches/patch{pid}_samples.h5"
            truth_dict = {c: truths[pid][c] for c in truths.dtype.names}
            title = title_fmt.format(**truth_dict)
            tag = title.replace(",", "_").lower().replace(" ", "").replace("\n", "")
            print(tag)

            tfig, ax = plot_trace(patchname)
            tfig.suptitle(title)
            tfig.tight_layout()
            tfig.savefig(f"figures/trace/{tag}_trace.png", dpi=200)
            pl.close(tfig)

            cfig, caxes = plot_corner(patchname)
            cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
            cfig.savefig(f"figures/corner/{tag}_corner.png", dpi=200)
            pl.close(cfig)

            rfig, raxes, val = plot_residual(patchname)
            vdict = deepcopy(truth_dict)
            vdict.update({c: val[c][0] for c in val.dtype.names})
            vtitle = "Last iteration:" + title_fmt.format(**vdict)
            rfig.suptitle(vtitle)
            rfig.savefig(f"figures/residuals/{tag}_residual.png", dpi=200)
            pl.close(rfig)

        sys.exit()

    else:
        bands = [str(b) for b in np.unique(truths["band"]).tolist()]
        scat = make_catalog(args.root, n_full=len(truths), bands=bands)
        fits.writeto(f"{args.root}/samples_catalog.fits", scat, overwrite=True)

    if args.parname:
        dfig, daxes = compare_parameters(scat, truths, args.parname,
                                         point_type=args.point_type)
        daxes[-1].legend(loc="upper left")
        dfig.savefig(f"figures/compare_{args.parname}_{args.point_type}.png", dpi=300)

    else:
        aflux, tflux = aperture_flux(scat, truths)
        sfig, saxes = snr_plot(aflux, tflux, truths, bands)
        sfig.savefig("figures/apflux_snr.png", dpi=200)

        ffig, faxes = pl.subplots(figsize=(8, 4))
        for b in bands:
            sel = truths["band"] == b
            faxes.plot(truths[sel]["snr"] * np.random.uniform(0.9, 1.1, sel.sum()),
                       aflux[sel].mean(axis=-1) * 2.0,
                       label=b, marker="o", linestyle="", alpha=0.5)
        faxes.axhline(1.0, color="k", linestyle=":")
        faxes.set_xlabel("SNR")
        faxes.set_ylabel("forcepho aperture flux (50th pctile) / true aperture flux")
        faxes.legend()
        faxes.set_xscale("log")
        ffig.tight_layout()
        ffig.savefig("figures/flux_comparison.png", dpi=200)