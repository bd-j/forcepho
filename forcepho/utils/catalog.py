# -*- coding: utf-8 -*-

import os
import numpy as np
from astropy.io import fits

__all__ = ["write_table", "out_dtype", "combine_chains",
           "pctile_cat"]


def write_table(out, cats, extnames=[], **header):

    full = fits.HDUList([fits.PrimaryHDU()] + [fits.BinTableHDU(cat) for cat in cats])
    for i, hdu in enumerate(full):
        for k, v in header.items():
            try:
                hdu.header[k] = v
            except:
                pass
    for i, ext in enumerate(extnames):
        full[i+1].header["EXTNAME"] = ext
    full.writeto(out, overwrite=True)
    return full


SHAPES = ["ra", "dec", "pa", "q", "sersic", "rhalf"]

def out_dtype(npoint=0, shapes=SHAPES, bands=[]):
    params = bands + shapes
    icols = [("id", "U30"), ("source_index", "<i4"), ("wall", "<f4"),
             ("lnp_best", "<f8")]  # ("filename")]
    if npoint > 3:
        icols += [("lnp", "<f8", npoint)]
    new = np.dtype(icols + [(c, float, npoint) for c in params])
    return new


def combine_chains(chaincat, bands, groups={}):
    """Take a chain cat and a dictionary specifying sources to be combined,
    and combine the fluxes for the chains.
    """
    hdr = {}
    scat = chaincat
    npoint = len(chaincat["ra"][0])
    dtype = out_dtype(npoint, shapes=["ra", "dec"], bands=bands)
    ocat = np.zeros(len(groups), dtype=dtype)

    mid = scat["id"].tolist()
    # loop over combined objects
    for i, (g, sub) in enumerate(groups.items()):
        ocat[i]["id"] = float(g)
        hdr[f"G{g}"] = ",".join([str(s) for s in sub])
        # get indices of components for this combination
        ids = [str(s) for s in sub]
        inds = [mid.index(i) for i in ids]
        # sum fluxes at each sample
        for b in bands:
            chain = scat[inds][b].sum(axis=0)
            ocat[i][b] = chain
        # average positions at each sample
        for c in ["ra", "dec"]:
            # straight mean, not a barycenter
            chain = scat[inds][c].mean(axis=0)
            ocat[i][c] = chain

    return ocat, hdr


def pctile_cat(samplecat, percentiles=[16, 50, 84]):
    """Make percentile based assymetric errorbars.

    Example shows how to plot asymmetric 1-sigma-ish errorbars:

    >>> scat = "path/to/postsample/catalog.fits"
    >>> ecat, hdr = make_errorbars(scat, percentiles=[16, 50, 84])
    >>> ecols = hdr["SMPLCOLS"].split(",")
    >>> colname = "rhalf"
    >>> y = ecat[colname][:,1]
    >>> e = np.diff(ecat[colname], axis=-1).T
    >>> ax.errorbar(y, y, e, linestyle="", marker="o", ecolor="k")

    Parameters
    ----------
    samplecat : string or structured ndarray
        Name of fits file contining result of
        :py:func:`forcepho.postprocess.postsample_catalog()`

    percentiles : list of floats in the interval (0, 100)
        The percentiles to compute

    Returns
    -------
    errorcat : structured ndarray
        Catalog of percentile values for each parameter.  These are given
        in the same order as the list in the `percentiles` keyword.

    hdr : dictionary or FITSHeader
        information about the errorbars.
    """
    if type(samplecat) is str:
        cat = np.array(fits.getdata(samplecat))
        hdr = fits.getheader(samplecat)
        bands = hdr["FILTERS"].split(",")
    else:
        cat = samplecat
        hdr = dict(PCTS=",".join([str(p) for p in percentiles]))

    desc, scol = [], []
    for d in cat.dtype.descr:
        if len(d) == 3:
            scol.append(d[0])
            desc.append((d[0], d[1], len(percentiles)))
        else:
            desc.append(d)
    ecat = np.zeros(len(cat), dtype=np.dtype(desc))
    for col in ecat.dtype.names:
        if col in scol:
            pct = np.percentile(cat[col], q=percentiles, axis=-1)
            ecat[col] = pct.T
        else:
            ecat[col] = cat[col]

    hdr["PCTS"] = ",".join([str(p) for p in percentiles])
    hdr["SMPLCOLS"] = ",".join(scol)

    return ecat, hdr


def rectify_jades(tablename, bands=["F200W"], pixscale=0.06):

    table = fits.getdata(tablename)
    pa = np.pi - np.deg2rad(table["PA"]) # need to convert from E of N to N of E
    sqrtq = np.sqrt(table["A"] / table["B"])
    rhalf = table["A"] * sqrtq * pixscale

    dt = sourcecat_dtype(bands=bands)
    cat = np.zeros(len(pa), dtype=dt)
    cat["ra"] = table["ra"]
    cat["dec"] = table["dec"]
    cat["pa"] = pa
    cat["q"] = sqrtq
    cat["rhalf"] = rhalf
    cat["id"] = table["ID"]
    for b in bands:
        cat[b] = table[f"{b}_CIRC2"]

    return cat