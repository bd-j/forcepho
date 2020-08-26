#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from astropy.io import fits
import h5py

from forcepho.sources import Galaxy
from forcepho.fitting import Result

__all__ = ["Logger",
           "rectify_catalog",
           "extract_block_diag",
           "make_result", "get_results",
           "make_statscat", "make_chaincat"]


class Logger:

    """A simple class that stores log information with similar API to logging.Logger
    """

    def __init__(self, name):
        self.name = name
        self.comments = []

    def info(self, message, timetag=None):
        if timetag is None:
            timetag = time.strftime("%y%b%d-%H.%M", time.localtime())

        self.comments.append((message, timetag))

    def serialize(self):
        log = "\n".join([c[0] for c in self.comments])
        return log


def sourcecat_dtype(source_type=np.float64, bands=[]):
    """Get a numpy.dtype object that describes the structured array
    that will hold the source parameters
    """
    nband = len(bands)
    tags = ["id", "source_index", "is_active", "is_valid", "n_iter", "n_patch"]

    dt = [(t, np.int32) for t in tags]
    dt += [(c, source_type)
           for c in Galaxy.SHAPE_COLS]
    dt += [(c, source_type)
           for c in bands]
    return np.dtype(dt)


def rectify_catalog(sourcecatfile, rhrange=(0.051, 0.29), qrange=(0.2, 0.99),
                    rotate=False, reverse=False):
    cat = fits.getdata(sourcecatfile)
    header = fits.getheader(sourcecatfile)
    bands = [b.strip() for b in header["FILTERS"].split(",")]

    n_sources = len(cat)
    cat_dtype = sourcecat_dtype(bands=bands)
    sourcecat = np.zeros(n_sources, dtype=cat_dtype)
    sourcecat["source_index"][:] = np.arange(n_sources)
    assert np.all([c in cat.dtype.names for c in Galaxy.SHAPE_COLS])
    for f in cat.dtype.names:
        if f in sourcecat.dtype.names:
            sourcecat[f][:] = cat[f][:]

    # --- Rectify shape columns ---
    sourcecat["rhalf"][:] = np.clip(sourcecat["rhalf"], *rhrange)
    sourcecat["q"][:] = np.clip(np.sqrt(sourcecat["q"]), *qrange)
    # rotate PA by +90 degrees but keep in the interval [-pi/2, pi/2]
    if rotate:
        p = sourcecat["pa"] > 0
        sourcecat["pa"] += np.pi / 2. - p * np.pi
    if reverse:
        sourcecat["pa"] *= -1.0

    return sourcecat, bands, header


def extract_block_diag(a, n, k=0):
    """Extract block diagonal elements from an array

    Parameters
    ----------
    a : ndarray, of shape (N, N)
        The input array

    n : int
        The size of each block

    Returns
    -------
    b : narray of shape (N//n, n, n)
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")
    if k > 0:
        a = a[:,n*k:]
    else:
        a = a[-n*k:]

    n_blocks = min(a.shape[0]//n, a.shape[1]//n)
    new_shape = (n_blocks, n, n)
    new_strides = (n*a.strides[0] + n*a.strides[1],
                   a.strides[0], a.strides[1])

    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)


def make_result(result, region, active, fixed, model,
                bounds=None, patchID=None, step=None, stats=None):
    """
    Parameters
    ----------
    result : a fitting.Result() object
        A namespace that contains fitting results

    region : a region.Region object
        The region defining the patch; its parameters will be added to the
        result.

    active : structured ndarray
        The active sources and their starting parameters.

    fixed : structured ndarray
        The fixed sources and their parameters.

    model : a model.PosteriorModel object
        Must contain `proposer.patch` and `scene` attributes

    bounds : optional
        If given, a structured ndarrray of lower and upper bounds for
        each parameter of each source.

    patchID : optional
        An integer giving the unique patch ID.

    step : optional
        If supplied, a littlemcmc NUTS step obect.  this contains the covariance matrix

    stats : optional
        If supplied, a littlemcmc stats object.

    Returns
    -------

    result : result container
        A simple namespace with numerous attributes added

    qcat : structured ndarray
        A structured array of the parameter values in the last sample of the chain.

    block_covs : ndarray of shape (N_source, N_param, N_param)
        The covariance matrices for the sampling potential, extracted as block
        diagonal elements of the full N_source * N_param square covariance
        array.  Not that it is in the units of the transformed, unconstrained
        sampling parameters.  If the prior bounds change, the covariance matrix
        is no longer valid (or must be retransformed)
    """

    patch = model.proposer.patch
    scene = model.scene
    bands = np.array(patch.bandlist, dtype="S")  # Bands actually present in patch
    shapenames = np.array(scene.sources[0].SHAPE_COLS, dtype="S")
    ref = np.array(patch.patch_reference_coordinates)
    block_covs = None

    out = result

    # --- Header ---
    out.patchID = patchID
    out.reference_coordinates = ref
    out.bandlist = bands
    out.shapenames = shapenames

    # --- region, active, fixed ---
    for k, v in region.__dict__.items():
        setattr(out, k, v)
    out.active = np.array(active)
    if fixed is not None:
        out.fixed = np.array(fixed)

    # --- chain and covariance ---
    # keep chain as a structured array? all info is saved to make it later
    #chaincat = make_chaincat(out.chain, bands, active, ref,
    #                         shapes=shapenames)
    if step is not None:
        try:
            out.cov = np.array(step.potential._cov.copy())
        except(AttributeError):
            out.cov = np.diag(step.potential._var.copy())
        if stats is not None:
            out.stats = make_statscat(stats, step)

        n_param = len(bands) + len(shapenames)
        block_covs = extract_block_diag(out.cov, n_param)

    # --- priors ---
    if bounds is not None:
        out.bounds = bounds

    # --- last position as structured array ---
    # FIXME: do this another way; maybe take a dtype from the superscene
    # or use make_chaincat
    qlast = out.chain[-1, :]
    scene.set_all_source_params(qlast)
    patch.unzerocoords(scene)
    for i, source in enumerate(scene.sources):
        source.id = active[i]["source_index"]
    qcat = scene.to_catalog(extra_cols=["source_index"])
    qcat["source_index"][:] = active["source_index"]
    out.final = qcat

    # qcat = active.copy()
    # for f in chaincat.dtype.names:
    #     if f in qcat.dtype.names:
    #         try:
    #             qcat[f][:] = chaincat[f][:, -1]
    #         except(IndexError):
    #             qcat[f][:] = chaincat[f][:]


    return out, qcat, block_covs


def get_results(fn):
    with h5py.File(fn, "r") as res:
        chain = res["chain"][:]
        #bands = res["bandlist"][:].astype("U").tolist()
        bands = ["Fclear"]
        ref = res["reference_coordinates"][:]
        active = res["active"][:]
        stats = res["stats"][:]

    cat = make_chaincat(chain, bands, active, ref)
    return cat, active, stats


def make_statscat(stats, step):
    # Reshape `stats` to an array
    dtype = np.dtype(list(step.stats_dtypes[0].items()))
    stats_arr = np.zeros(len(stats), dtype=dtype)
    for c in stats_arr.dtype.names:
        stats_arr[c][:] = np.array([s[c] for s in stats])
    return stats_arr


def make_chaincat(chain, bands, active, ref, shapes=Galaxy.SHAPE_COLS):
    # --- Get sizes of things ----
    n_iter, n_param = chain.shape
    n_band = len(bands)

    n_param_per_source = n_band + len(shapes)
    assert (np.mod(n_param, n_param_per_source) == 0)
    n_source = int(n_param / n_param_per_source)
    assert (n_source == len(active))

    # --- generate dtype ---
    colnames = bands + shapes
    cols = [("source_index", np.int32)] + [(c, np.float64, (n_iter,))
                                           for c in colnames]
    dtype = np.dtype(cols)

    # --- make and fill catalog
    cat = np.zeros(n_source, dtype=dtype)
    cat["source_index"][:] = active["source_index"]
    for s in range(n_source):
        for j, col in enumerate(colnames):
            cat[s][col] = chain[:, s * n_param_per_source + j]

    # rectify parameters
    cat["ra"] += ref[0]
    cat["dec"] += ref[1]

    return cat
