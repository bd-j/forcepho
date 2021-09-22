#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
try:
    import matplotlib.pyplot as pl
    from matplotlib.backends.backend_pdf import PdfPages
except:
    pass

import json
from astropy.io import fits

from .utils import read_config
from .patches.storage import MetaStore
from .superscene import isophotal_radius
from .fitting import Result


__all__ = ["Residuals", "Samples",
           "postop_catalog", "postsample_catalog",
           "cat_to_reg", "write_sourcereg", "write_patchreg",
           "write_images",
           "residual_pdf", "chain_pdf"]


class Residuals:

    def __init__(self, filename):
        self.filename = filename
        self.handle = h5py.File(self.filename, "r")
        self.exposures = self.handle["epaths"][:]
        self.reference_coordinates = self.handle["reference_coordinates"][:]

    def show(self, e=0, exp="", axes=[], **plot_kwargs):
        if not exp:
            exp = self.exposures[e]
        xpix, ypix = self.handle[exp]["xpix"][:], self.handle[exp]["ypix"][:]
        data = self.handle[exp]["data"][:]
        resid = self.handle[exp]["residual"][:]
        model = data - resid

        # keep the scales the same
        kwargs = dict(vmin=min(data.min(), model.min(), resid.min()),
                      vmax=max(data.max(), model.max(), resid.max()))
        kwargs.update(**plot_kwargs)
        _ = show_exp(xpix, ypix, data, ax=axes.flat[0], **kwargs)
        _ = show_exp(xpix, ypix, resid, ax=axes.flat[1], **kwargs)
        _ = show_exp(xpix, ypix, model, ax=axes.flat[2], **kwargs)

        return data, model, resid

    def mark_sources(self, ra, dec, e=0, exp="", axes=[],
                     plot_kwargs=dict(marker="x", linestyle="", color="red"),
                     **extras):

        apix = self.sky_to_pix(ra, dec, e, exp)

        plot_kwargs.update(extras)
        axes = np.atleast_1d(axes)
        [ax.plot(apix[:, 0], apix[:, 1], **plot_kwargs) for ax in axes]

        return axes

    def sky_to_pix(self, ra, dec, e=0, exp=""):
        if not exp:
            exp = self.exposures[e]
        ee = self.handle[exp]
        pix = sky_to_pix(ra, dec, ee, ref_coords=self.reference_coordinates)
        return pix

    def fill_images(self, images={}, fill_type="residual", imshape=(2048, 2048)):
        for e in self.exposures:
            if e not in images:
                images[e] = np.zeros(imshape) + np.nan
            xpix, ypix = self.handle[e]["xpix"][:].astype(int), self.handle[e]["ypix"][:].astype(int)
            arr = self.handle[e][fill_type][:]
            images[e][xpix, ypix] = arr

        return images


class Samples(Result):
    """Just an alias for the Result class
    """
    def __init__(self, filename):
        super(Samples, self).__init__(filename)


def run_metadata(root):
    with open(f"{root}/config.json") as f:
        config = json.load(f)
    scenestr = config["scene_catalog"].replace(".fits", "")

    with open(f"{root}/{scenestr}_log.json", "r") as f:
        logs = json.load(f)
        slog = logs["sourcelog"]
        plog = logs["patchlog"]
    final = fits.getdata(f"{root}/{scenestr}.fits")

    return config, plog, slog, final


def postop_catalog(root, bands=None, catname=None):
    """Make an input catalog from the post-optimization catalog.
    """
    samples = Samples(f"{root}/patches/patch0_samples.h5")
    if catname is None:
        config = json.loads(samples.config)
        #config = read_config(config)
        raw = config["raw_catalog"]
        catname = raw.replace(".fits", "_postop.fits")
    if bands is None:
        bands = samples.bands

    catalog = glob.glob(f"{root}/*fits")[0]
    hdus = fits.open(catalog)
    cat = hdus[1].data
    cat["n_iter"] = 0
    cat["n_patch"] = 0
    # oof
    hdus[0].header["POSTOP"] = True
    hdus[0].header["FILTERS"] = ",".join(bands)
    hdus.writeto(catname, overwrite=True)
    print("Remember units of q are now sqrt(b/a)")


def postsample_catalog(root, catname=None):
    # get summary info
    config, plog, slog, final = run_metadata(root)
    patches = plog

    # Get catalog data type
    s0 = Samples(f"{root}/patches/patch{plog[0]}_samples.h5")
    desc = s0.chaincat.dtype.descr
    bands = s0.bands
    new = [("id", "<i4"), ("source_index", "<i4"), ("patch_id", "<i4")]
    vcol = []
    for d in desc:
        if len(d) == 3:
            # Only keep real samples
            col = (d[0], d[1], (s0.n_sample,))
            vcol.append(d[0])
            new.append(col)

    # Make and fill the catalog
    cat = np.zeros(len(final), np.dtype(new))
    cat["id"] = -1
    for p in patches:
        s = Samples(f"{root}/patches/patch{p}_samples.h5")
        inds = s.chaincat["source_index"]
        cat["id"][inds] = s.active["id"]
        cat["patch_id"][inds] = p
        cat["source_index"][inds] = inds
        for col in vcol:
            cat[col][inds] = s.chaincat[col][:, -s0.n_sample:]

    cat = cat[cat["id"] >= 0]

    # write catalog
    if catname is not None:
        # oof
        table = fits.BinTableHDU.from_columns(cat)
        hdr = fits.Header()
        hdr["FILTERS"] = ",".join(bands)
        hdr["ROOT"] = root
        full = fits.HDUList([fits.PrimaryHDU(header=hdr), table])
        full.writeto(catname, overwrite=True)
    return cat


def find_multipatch(root):
    """Load all the sourceIDs, patchIDs, and chains for sources that appeared in
    multiple patches.

    Returns
    -------
    sources : list of ints
        The sources that appeared in multiple patches.  Length (n_dupe,)

    patches : list of lists
        A list of the patches that each source appeared appeared in, length (n_dupe)

    chains : list of ndarrays
        A list of the posterior sample chains for each source.
    """
    config, plog, slog, final = run_metadata(root)
    samples = [Samples(f"{root}/patches/patch{p}_samples.h5") for p in plog]
    chains, sources, patches = [], [], []
    for sid, pids in slog.items():
        if len(pids) == 1:
            continue
        c = []
        for pid in pids:
            ai = samples[int(pid)].active["source_index"].tolist().index(int(sid))
            c.append(samples[int(pid)].chaincat[ai:ai+1])
        chains.append(np.concatenate(c))
        sources.append(sid)
        patches.append(pids)

    return sources, patches, chains


def check_multipatch(root, n_sample=256):
    sid, pids, chains = find_multipatch(root)
    stats = {}
    for s, p, c in zip(sid, pids, chains):
        stats[s] = {}
        for col in vcols:
            sm = c[col][:, -n_sample:]
            stats[s][col] = sm.mean(axis=-1), sm.std(axis=-1)
    return stats


def make_regions(cat, roi=None, ellipse=False):
    from astropy import units as u
    from regions import EllipseSkyRegion, Regions

    if roi is None:
        roi = cat["rhalf"]

    regs = []
    for i, row in enumerate(cat):
        center_sky = SkyCoord(row["ra"], row["dec"], unit='deg', frame='fk5')

        if ellipse:
            sqrtq = row["q"]
            pa = np.rad2deg(-row["pa"])
            #pa = np.rad2deg(row["pa"])
        else:
            sqrtq = 1
            pa = 0.0
        a = roi[i] / sqrtq
        b = sqrtq * roi[i]
        reg = EllipseSkyRegion(center=center_sky, height=b * u.arcsec,
                               width=a * u.arcsec, angle=pa * u.deg)
        regs.append(reg)

    return Regions(regs)


def cat_to_reg(cat, slist="", showid=False, default_color="green",
               valid=None, roi=None, ellipse=False):

    if type(cat) is str:
        from astropy.io import fits
        cat = fits.getdata(cat)

    regions = make_regions(cat, roi=roi, ellipse=ellipse)

    if valid is not None:
        for i, r in enumerate(regions):
            if valid[i]:
                r.visual["color"] = "green"
            else:
                r.visual["color"] = "red"

    if showid:
        for i, r in enumerate(regions):
            r.meta["text"] = f"{cat[i]['id']}"

    if slist:
        regions.write(slist, format="ds9", overwrite=True)

    return regions


def write_sourcereg(root, slist="sources.reg", showid=False,
                    isophote=("F160W", 0.1/(0.06**2))):
    """
    Parameters
    ----------
    isophote : tuple of string, float (optional)
        If given, the band for which to compute the isophotal radius, and the isophote level in nJy/arcsec^2
    """
    f = glob.glob(f"{root}/outscene*.fits")[0]
    os.makedirs(f"{root}/image", exist_ok=True)
    rfile = f"{root}/image/{slist}"

    cat = fits.getdata(f)
    valid = cat["n_iter"] > 0
    if isophote is not None:
        roi = isophotal_radius(isophote[1], cat[isophote[0]], cat["rhalf"], sersic=cat["sersic"])
        roi[np.isnan(roi)] = cat["rhalf"][np.isnan(roi)]
    else:
        roi = cat["rhalf"]
    regions = cat_to_reg(cat, slist=rfile, showid=showid, roi=roi, ellipse=True, valid=valid)
    return regions


def write_patchreg(root, plist="patches.reg"):
    files = glob.glob(f"{root}/patches/*samples.h5")
    assert len(files) > 0
    patches = range(len(files))

    with open(f"{root}/image/{plist}", "w") as out:
        out.write("global color=blue\n")
        fmt = 'fk5;circle({}d,{}d,{}d) # text="{}"\n'
        for p in patches:
            s = Samples(f"{root}/patches/patch{p}_samples.h5")
            out.write(fmt.format(s.region.ra, s.region.dec, s.region.radius, p))


def write_images(root, subdir="image", metafile=None, show_model=False, show_chi=False):

    files = glob.glob(f"{root}/patches/*residuals.h5")
    assert len(files) > 0

    print(f"Writing to {root}/{subdir}")
    os.makedirs(f"{root}/{subdir}", exist_ok=True)

    deltas, datas, ierrs = {}, {}, {}
    patches = range(len(files))
    for p in patches:
        r = Residuals(f"{root}/patches/patch{p}_residuals.h5")
        r.fill_images(deltas, fill_type="residual")
        r.fill_images(datas, fill_type="data")
        r.fill_images(ierrs, fill_type="ierr")

    if metafile:
        print(f"Using {metafile}")
        metastore = MetaStore(metafile)

    stypes = ["data", "delta", "model", "residual"]

    imnames = list(datas.keys())
    for n in imnames:
        band, exp = n.decode("utf-8").split("/")
        show = [datas[n], deltas[n], None, None]
        if show_model:
            show[2] = datas[n] - deltas[n]
        if show_chi:
            show[3] = deltas[n] * ierrs[n]
        # should add ability to put a real header here
        # also determine actual image sizes instead of assuming 2048^2
        # could be done using the metastore
        for i, s in enumerate(show):
            if s is None:
                continue
            stype = stypes[i]
            sh = fits.PrimaryHDU(s.T)
            if metafile:
                hdr = metastore.headers[band][exp]
                hdr["BUNIT"] = "nJy"
                assert sh.header["NAXIS"] == hdr["NAXIS"]
                assert sh.header["NAXIS1"] == hdr["NAXIS1"]
                assert sh.header["NAXIS2"] == hdr["NAXIS2"]
                sh.header.update(hdr)
                sh.header["IMTYPE"] = stype.upper()
            sh.writeto(f"{root}/{subdir}/{exp}_{stype}.fits", overwrite=True)


def residual_pdf(root="opt_all_v0.5_full", n=3, e=3):
    """
    Parameters
    ----------
    root : string
        base directory where patches are located

    n : int, optional (default 3)
        number of patches to show per page

    e : int
        exposure number to show; i.e. this will show Residuals.exposures[e]
    """
    npatch = len(glob.glob(f"{root}/patches/*samples*h5"))
    patches = range(npatch)
    J = npatch // n

    pdf = PdfPages(f"{root}/{os.path.basename(root)}_residuals_{e}.pdf")
    for j in range(J):
        fig, ax = pl.subplots(n, 3, sharex="row", sharey="row")
        for i, p in enumerate(patches[n*j:n*(j+1)]):
            residuals = Residuals(f"{root}/patches/patch{p}_residuals.h5")
            samples = Samples(f"{root}/patches/patch{p}_samples.h5")
            try:
                residuals.show(e, axes=ax[i, [0, 2, 1]], vmin=-0.5, vmax=0.5)
            #residuals.mark_sources(samples.active["ra"], samples.active["dec"],
            #                       e, axes=[ax[i, 0]], color="magenta")
                print(samples.message, samples.ncall)
            #print(samples.active[0]["source_index"])
                if hasattr(samples, "fixed"):
                    residuals.mark_sources(samples.fixed["ra"], samples.fixed["dec"],
                                           e, axes=[ax[i, -1]], color="red")
            except(IndexError):
                pass

        pdf.savefig(fig, dpi=500)
        pl.close(fig)
    pdf.close()


def chain_pdf(samples, fn="./chain.pdf", dh=1.0):
    """Make a PDF of the posterior chains for every object in a patch

    Parameters
    ----------
    samples: forcepho.reconstruction.Samples instance
        The posterior samples object for a given patch with chain information.
    """
    ns = len(samples.active)
    nb = len(samples.bands)
    pdf = PdfPages(fn)
    for i in range(ns):
        tuning = max(np.where(samples.stats["tune"])[0])
        act = samples.active[i]
        ncol, nrow = 2, max(7, nb-1)
        fs = (nrow * dh * 7./5., nrow * dh)
        fig, axes = pl.subplots(nrow, ncol, figsize=fs, sharex=True)
        samples.show_chain(i, bandlist=samples.bands[0:1], axes=axes[:7, 0], truth=act)
        samples.show_chain(i, bandlist=samples.bands[1:], show_shapes=False,
                           axes=axes[:(nb-1), 1], truth=act)
        if tuning > 1:
            [a.axvspan(0, tuning, alpha=0.3, color="gray", zorder=-1)
             for a in axes.flat]
        fig.suptitle(f"source ID={samples.active[i]['id']}")
        pdf.savefig(fig)
        pl.close(fig)
    pdf.close()


def show_precision(s, i, ax, nsigma=3, snr_max=10):
    from forcepho.superscene import flux_bounds
    flux = s.final[s.bands[i]]
    sigma = np.sqrt(np.diag(np.linalg.pinv(s.precisions[i])))
    sigma = np.maximum(sigma, flux/snr_max)
    oo = np.argsort(flux)
    lo, hi = flux_bounds(flux, nsigma)
    ax.fill_between(flux[oo], lo[oo], hi[oo], alpha=0.5)
    ax.fill_between(flux[oo], (flux - nsigma * sigma)[oo], (flux + nsigma * sigma)[oo], alpha=0.5)


if __name__ == "__main__":

    modes = ["images", "patches", "catalog", "chains", "postop"]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="images",
                        choices=modes)
    parser.add_argument("--root", type=str, default="../output/opt_all_v0.5_linopt_debug_v4")
    parser.add_argument("--metafile", type=str, default="../data/stores/meta_hlf2_udf.json")
    parser.add_argument("--exp", type=int, default=14)
    parser.add_argument("--catname", type=str, default=None)
    args = parser.parse_args()

    #write_sourcereg(args.root, slist="sources.reg", showid=True)
    #write_patchreg(args.root, plist="patches.reg")

    if args.mode == 'images':
        write_images(args.root, metafile=args.metafile, show_model=True)
        write_patchreg(args.root)
        write_sourcereg(args.root, showid=True)

    elif args.mode == 'patches':
        residual_pdf(root=args.root, e=args.exp)

    elif args.mode == "chains":
        patches = glob.glob(f"{args.root}/patches/*samples*h5")
        for p in patches:
            chain_pdf(Samples(p), p.replace("samples.h5", "chain.pdf"))

    elif args.mode == "postop":
        print(f"writing to {args.catname}")
        postop_catalog(args.root, catname=args.catname)

    elif args.mode == "catalog":
        print(f"writing to {args.catname}")
        cat = postsample_catalog(args.root, catname=args.catname)


    else:
        print(f"{args.mode} not a valid mode.  choose one of {modes}")
