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
import h5py

from .fitting import Result
from .utils import sky_to_pix, populate_image, isophotal_radius
from .patches.storage import MetaStore
from .superscene import flux_bounds

__all__ = ["Residuals", "Samples",
           "run_metadata",
           "postop_catalog", "postsample_catalog",
           "make_errorbars", "flux_unc_linear",
           "write_sourcereg", "write_patchreg",
           "write_images", "show_exp",
           "residual_pdf", "chain_pdf",
           "combined_rhalf"]


class Residuals:
    """Structure for reading and storing residual data for a single patch from
    HDF5 files generated by forcepho.utils.write_residuals.  Includes
    convenience methods for plotting residuals or placing them within larger
    original images.
    """

    def __init__(self, filename):
        self.filename = filename
        self.handle = h5py.File(self.filename, "r")
        self.exposures = [e.decode("utf-8") for e in self.handle["epaths"][:]]
        self.ebands = [e.decode("utf-8") for e in self.handle["ebands"][:]]
        self.reference_coordinates = self.handle["reference_coordinates"][:]

    def make_exp(self, e=0, exp=None, value="data"):
        if not exp:
            exp = self.exposures[e]
        xpix, ypix = self.handle[exp]["xpix"][:], self.handle[exp]["ypix"][:]
        data = self.handle[exp][value][:]

        im, lo, hi = populate_image(xpix, ypix, data)

        return im, lo, hi

    def show(self, e=0, exp="", axes=[], **plot_kwargs):
        if not exp:
            exp = self.exposures[e]
        xpix, ypix = self.handle[exp]["xpix"][:], self.handle[exp]["ypix"][:]
        data = self.handle[exp]["data"][:]
        ierr = self.handle[exp]["ierr"][:]
        resid = self.handle[exp]["residual"][:]
        model = data - resid
        chi = data * ierr
        resid[ierr == 0] = np.nan
        chi[ierr == 0] = np.nan

        # keep the scales the same
        kwargs = dict(vmin=min(data.min(), model.min(), resid.min()),
                      vmax=max(data.max(), model.max(), resid.max()))
        kwargs.update(**plot_kwargs)
        # could replace this with calls to make_exp and imshow
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

    def write_cutouts(self, bands, cutdir=".", target="cool",
                      as_fits=True, **imkwargs):

        raise NotImplementedError

        count = {b: 1 for b in bands}
        for exp in self.exposures:
            #imlabel = "_".join(os.path.basename(exp).split("_")[1:4])
            #name = exp.replace("/home/bjohnso6/smacs/pho/output/",
            #                "/Users/bjohnson/Projects/smacs/pho/output/gpu/")
            #hdr = fits.getheader(name)
            band = hdr["FILTER"]
            det = hdr["DETECTOR"]
            e = hdr["EXPOSURE"]
            kinds = ["data", "model", "residual"]
            for i, k in enumerate(kinds):
                arr, lo, hi = self.make_exp(exp=exp, value=k)
                num = count[band]
                out = f"{cutdir}/{target}_{band}_{num}_{kinds[i]}"
                if as_fits:
                    fits.writeto(f"{out}.fits", arr, overwrite=True, header=hdr)
                else:
                    pl.imsave(f"{out}.png", arr, **imkwargs)
            count[band] += 1

        return count

    def sky_to_pix(self, ra, dec, e=0, exp=""):
        if not exp:
            exp = self.exposures[e]
        ee = self.handle[exp]
        pix = sky_to_pix(ra, dec, ee, ref_coords=self.reference_coordinates)
        return pix

    def fill_images(self, images={}, headers={}, fill_type="residual",
                    metastore=None, imshape=(2048, 2048)):
        """Add the stored pixel data to the images in the supplied `images` dictionary.
        """
        for e in self.exposures:
            # create the empty array
            if e not in images:
                if metastore is not None:
                    band, exp = e.split("/")
                    hdr = metastore.headers[band][exp]
                    imshape = (hdr["NAXIS1"], hdr["NAXIS2"])
                    headers[e] = hdr
                elif os.path.exists(e):
                    hdr = fits.getheader(e)
                    imshape = (hdr["NAXIS1"], hdr["NAXIS2"])
                    headers[e] = hdr
                images[e] = np.zeros(imshape) + np.nan
            # Fill array with values for this patch
            xpix, ypix = self.handle[e]["xpix"][:].astype(int), self.handle[e]["ypix"][:].astype(int)
            arr = self.handle[e][fill_type][:]
            images[e][xpix, ypix] = arr

        return images, headers


class Samples(Result):
    """An alias for the forcepho.fitting.Result class
    """
    def __init__(self, filename):
        super(Samples, self).__init__(filename)

    def summary_dtype(self, npoint=0):
        bands, shapes, n_sample = list(self.bands), list(self.shape_cols), self.n_sample
        params = bands + shapes
        icols = [("id", "<f8"), ("source_index", "<i4"), ("wall", "<f4"),
                ("lnp_best", "<f8")] #("filename")]

        if npoint:
            n_sample = npoint
        else:
            icols += [("lnp", "<f8", n_sample)]

        new = np.dtype(icols + [(c, float, n_sample) for c in params])
        return new


def run_metadata(root):
    """
    Returns
    -------
    config : dict
        Configureation for the run as a dictionary

    plog : list of int
        The order of the patches

    slog : dict
        The sources in each patch.  Keyed by patch integer, values are a list of
        source indices

    final : structured ndarray
        The final state catalog for the superscene
    """
    with open(f"{root}/config.json") as f:
        config = json.load(f)
    scenestr = config.get("scene_catalog", "superscene.fits").replace(".fits", "")

    with open(f"{root}/{scenestr}_log.json", "r") as f:
        logs = json.load(f)
        slog = logs["sourcelog"]
        plog = logs["patchlog"]
    final = fits.getdata(f"{root}/{scenestr}.fits")

    return config, plog, slog, final


def postop_catalog(root, bands=None, catname=None):
    """Make an input catalog from the post-optimization catalog.  This catalog
    will be suitable for use as the inital catalog for a sampling run

    Also attemps to make a catalog of flux uncertainties in the 2nd extension.
    These are based on precision matrices if available.

    Parameters
    ----------
    root : string
        Name of the directory containing the optimization results

    bands : list of strings
        Name of the bands to include in the postop catalog.

    catname : string
        Name of the output catalog
    """
    samples = Samples(f"{root}/patches/patch0_samples.h5")
    if catname is None:
        config = json.loads(samples.config)
        raw = config["raw_catalog"]
        catname = raw.replace(".fits", "_postop.fits")
    if bands is None:
        bands = samples.bands

    catalog = glob.glob(f"{root}/*fits")[0]
    hdus = fits.open(catalog)
    cat = hdus[1].data
    cat["n_iter"] = 0
    cat["n_patch"] = 0

    try:
        unc = flux_unc_linear(root)
        hdus.append(fits.BinTableHDU(unc))
        hdus[0].header["EXT2"] = "flux uncertainties"
    except(AttributeError):
        pass

    # oof
    hdus[0].header["EXT1"] = "maximum-a-posteriori"
    hdus[0].header["POSTOP"] = True
    hdus[0].header["FILTERS"] = ",".join(bands)
    hdus.writeto(catname, overwrite=True)
    print("Remember units of q are now sqrt(b/a)")


def postsample_catalog(root, catname=None, patches=None):
    """Make a catalog of posterior samples for each parameter, combining all
    patches in a given run.

    Parameters
    ----------
    root : string
        Name of the directory containing the optimization results

    patches : list of ints

    Returns
    -------
    cat : structured ndarray
        A structured array of posterior samples, of shape (n_source,).  Each row
        corresponds to a different input source, and has fields for the
        parameters of that source each with shape (n_sample,)
    """
    # get summary info
    if patches is None:
        config, plog, slog, final = run_metadata(root)
        patches = plog
    else:
        final = patches

    # Get catalog data type
    s = Samples(f"{root}/patches/patch{patches[0]}_samples.h5")
    bands, shapes, n_sample = list(s.bands), list(s.shape_cols), s.n_sample
    params = bands + shapes
    icols = [("id", "<i4"), ("source_index", "<i4"), ("patch_id", "<i4"),
             ("wall", "<f4"), ("lnp", "<f8", n_sample)]
    new = np.dtype(icols + [(c, float, n_sample) for c in params])

    # Make and fill the catalog
    cat = np.zeros(len(final), np.dtype(new))
    cat["id"] = -1
    for p in patches:
        s = Samples(f"{root}/patches/patch{p}_samples.h5")
        inds = s.chaincat["source_index"]
        cat["id"][inds] = s.active["id"]
        cat["patch_id"][inds] = p
        cat["source_index"][inds] = inds
        cat["wall"][inds] = s.wall_time
        cat["lnp"][inds] = s.stats["model_logp"][-n_sample:]
        for col in params:
            cat[col][inds] = s.chaincat[col][:, -n_sample:]

    #cat = cat[cat["id"] >= 0]

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


def flux_unc_linear(root, snr_max=1000):
    """Get flux uncertainties from the precision matrix of linear fits, put them
    in a catalog matching the parameter catalog line-by-line.  If precision matrix does not exist, use a simple default

    Parameters
    ----------
    root : string
        Directory location.  Must contain a set of files matching the pattern
        `<root>/patches/patch[1-9999]_samples.h5`

    snr_max : float, optional (default: 1000)
        Make sure the reprted uncertainties are large enough that this S/N is not exceeded

    Returns
    -------
    unc : structured ndarray
        A structured array of shape (n_active,) with the flux encertainties for
        each source as fields.  The dtype is the same as the 'active' structured
        array
    """
    config, plog, slog, final = run_metadata(root)
    patches = plog
    unc = np.zeros(len(final), dtype=final.dtype)
    unc["id"][:] = final["id"]
    unc["source_index"][:] = final["source_index"]

    # fill with default
    for b in config["bandlist"]:
        u = np.array(flux_bounds(final[b], 1)) - final[b]
        unc[b][:] = np.max(np.abs(u), axis=0)

    # now fill with precision matrix based results.
    for p in patches:
        patchfile = f"{root}/patches/patch{p}_samples.h5"
        s = Samples(patchfile)
        inds = s.chaincat["source_index"]

        if not hasattr(s, "precisions"):
            print(f"flux_unc_linear: No precisons for {patchfile}")

        for i, b in enumerate(s.bands):
            flux = s.final[b]
            Sigma = np.linalg.pinv(s.precisions[i])
            sigma = np.maximum(np.sqrt(np.diag(Sigma)), flux / snr_max)
            unc[b][inds] = sigma

    return unc


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


def write_sourcereg(root, slist="sources.reg", showid=False,
                    isophote=None):
    """
    Parameters
    ----------
    isophote : tuple of string, float (optional)
        If given, the band for which to compute the isophotal radius, and the
        isophote level in nJy/arcsec^2
    """
    f = glob.glob(f"{root}/outscene*.fits")[0]
    os.makedirs(f"{root}/image", exist_ok=True)
    rfile = f"{root}/image/{slist}"

    cat = fits.getdata(f)
    if isophote is not None:
        roi = isophotal_radius(isophote[1], cat[isophote[0]],
                               cat["rhalf"], sersic=cat["sersic"])
        roi[np.isnan(roi)] = cat["rhalf"][np.isnan(roi)]
    else:
        roi = cat["rhalf"]
    regions = cat_to_reg(cat, roi=roi, ellipse=True)
    regions.write(slist, format="ds9", overwrite=True)

    return regions


def write_patchreg(patchlist, plist="./patches.reg"):
    """
    patchlist : list of str
        The names of the _samples.h5 files constituting the patches
    """
    patchlist.sort()

    with open(plist, "w") as out:
        out.write("global color=blue\n")
        fmt = 'fk5;circle({}d,{}d,{}d) # text="{}"\n'
        for i, p in enumerate(patchlist):
            s = Samples(p)
            out.write(fmt.format(s.region.ra, s.region.dec, s.region.radius, i))


def write_images(root, subdir="image", metafile=None, show_model=False, show_chi=False):
    """Make data, residual, and optionally model images for the last iteration
    of the chain.
    """

    files = glob.glob(os.path.join(root, "patches/*residuals.h5"))
    assert len(files) > 0
    files.sort()

    print(f"Writing to {root}/{subdir}")
    os.makedirs(f"{root}/{subdir}", exist_ok=True)

    if metafile:
        print(f"Using {metafile}")
        metastore = MetaStore(metafile)
    else:
        metastore = None

    deltas, datas, ierrs, hdrs = {}, {}, {}, {}
    patches = range(len(files))
    for p in files:
        r = Residuals(p)
        r.fill_images(deltas, fill_type="residual", metastore=metastore)
        r.fill_images(datas, headers=hdrs, fill_type="data", metastore=metastore)
        r.fill_images(ierrs, fill_type="ierr", metastore=metastore)

    stypes = ["data", "delta", "model", "chi"]

    imnames = list(datas.keys())
    for n in imnames:
        exp = n.split("/")[-1].replace(".fits", "")
        hdr = hdrs.get(n, None)
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
            if hdr is not None:
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


def show_exp(xpix, ypix, value, ax=None, **imshow_kwargs):
    """Create a rectangular image that bounds the given pixel coordinates
    and assign `value` to the correct pixels. Pixels in the rectangle that do
    not have assigned values are given nan.  use imshow to display the image in
    standard astro format (x increasing left to right, y increasing bottom to
    top)
    """
    im, lo, hi = populate_image(xpix, ypix, value)

    ax.imshow(im.T, origin="lower",
              extent=(lo[0], hi[0], lo[1], hi[1]),
              **imshow_kwargs)
    return im


def show_precision(s, i, ax, nsigma=3, snr_max=10):
    from forcepho.superscene import flux_bounds
    flux = s.final[s.bands[i]]
    sigma = np.sqrt(np.diag(np.linalg.pinv(s.precisions[i])))
    sigma = np.maximum(sigma, flux/snr_max)
    oo = np.argsort(flux)
    lo, hi = flux_bounds(flux, nsigma)
    ax.fill_between(flux[oo], lo[oo], hi[oo], alpha=0.5)
    ax.fill_between(flux[oo], (flux - nsigma * sigma)[oo], (flux + nsigma * sigma)[oo], alpha=0.5)


def combined_rhalf(samples, stamp, band, sources=slice(None), step=5):

    stop = samples.chain.shape[0]
    start = stop - samples.n_sample

    rhalf = []

    inds = range(start, stop, step)
    for i in inds:
        cat = samples.get_sample_cat(i)
        im, scene = forcepho_slow_model(cat[sources], stamp, band)
        x, y = (stamp.xpix * im).sum() / im.sum(), (stamp.ypix * im).sum() / im.sum()
        r = np.hypot(stamp.xpix - x, stamp.ypix - y).flatten()
        f = im.flatten()
        oo = np.argsort(r)
        fo = np.cumsum(f[oo]) / im.sum()
        rhalf.append(np.interp(0.5, fo, r[oo]))

    return np.array(rhalf)


def forcepho_slow_model(cat, stamp, band, psf=None,
                        splinedata="../data/stores/sersic_splinedata_large.h5"):

    """Render a scene (given as a fpho catalog) onto a stamp using the slow
    code.  This can be useful to render scenes with idealized (narrow) PSFs.
    """

    from .sources import Galaxy
    from .slow.psf import PointSpreadFunction

    if psf is None:
        psf = PointSpreadFunction()
    stamp.psf = psf
    #stamp.psf.covariances = 4

    im = np.zeros([stamp.nx, stamp.ny])
    scene = []

    for catrow in cat:
        galaxy = Galaxy(splinedata=splinedata)
        #print(catrow[band])
        galaxy.from_catalog_row(catrow, filternames=[band])
        scene.append(galaxy)
        #print(galaxy.ra, galaxy.dec)
        #print(stamp.sky_to_pix(np.array([galaxy.ra, galaxy.dec])))

        i, g = stamp.render(galaxy, compute_deriv=False)
        #print(i.max())
        im += i.reshape(stamp.nx, stamp.ny)

    return im, scene


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
