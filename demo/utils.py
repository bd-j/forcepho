#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from astropy.io import fits

from forcepho.sources import Galaxy
from forcepho.fitting import Result


class Logger:

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

    n_param_per_source = n_band + 6
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


def make_result(result, region, active, fixed, model,
                patchID=None, step=None, stats=None):

    patch = model.proposer.patch
    scene = model.scene
    bands = np.array(patch.bandlist, dtype="S")  # Bands actually present in patch
    shapenames = np.array(scene.sources[0].SHAPE_COLS, dtype="S")
    ref = np.array(patch.patch_reference_coordinates)
    mass_matrix = None

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
    if step is not None:
        out.cov = np.array(step.potential._cov.copy())
        if stats is not None:
            out.stats = make_statscat(stats, step)
    # keep chain as a structured array? all info is saved to make it later
    #chaincat = make_chaincat(out.chain, bands, active, ref,
    #                         shapes=shapenames)

    # --- priors ---
    if model.transform is not None:
        out.upper_bound = np.array(model.transform.upper.copy())
        out.lower_bound = np.array(model.transform.lower.copy())

    # --- last position ---
    # FIXME: do this another way
    qlast = out.chain[-1, :]
    scene.set_all_source_params(qlast)
    patch.unzerocoords(scene)
    for i, source in enumerate(scene.sources):
        source.id = active[i]["source_index"]
    qcat = scene.to_catalog(extra_cols=["source_index"])
    qcat["source_index"][:] = active["source_index"]
    out.final = qcat

    return out, qcat, mass_matrix


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
