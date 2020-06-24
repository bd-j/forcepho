#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from astropy.io import fits

from forcepho.sources import Galaxy


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
    cols = [("source_index", np.int)] + [(c, np.float, (n_iter,))
                                         for c in colnames + aper_bands]
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


def dump(region, active, fixed, model, chain,
         step=None, stats=None, start=None):
    """Save the results to an HDF file
    """
    patch = model.proposer.patch
    scene = model.scene
    bands = np.array(patch.bandlist, dtype="U")  # Bands actually present in patch
    shapenames = np.array(scene.sources[0].SHAPE_COLS, dtype="U")
    ref = np.array(patch.patch_reference_coordinates)
    mass_matrix = None

    with h5py.File(filename, "w") as out:

        # --- Header ---
        out.attrs["patchID"] = patchID
        out.attrs["reference_coordinates"] = ref
        out.attrs["bandlist"] = bands
        out.attrs["shapenames"] = shapenames

        # --- region, active, fixed ---
        for k, v in region.__dict__.items():
            out.attrs[k] = v
        out.create_dataset("active", data=np.array(active))
        if fixed is not None:
            out.create_dataset("fixed", data=np.array(fixed))

        # --- chain and covariance ---
        out.create_dataset("chain", chain)
        if start is not None:
            out.create_dataset("start", data=np.array(start))
        if step is not None:
            out.create_dataset("cov", data=np.array(step._cov))
        # keep chain as a structured array?
        # all infor is saved to make it
        #chaincat = make_chaincat(chain, bands, active, ref,
        #                         shapes=shapenames)

        # --- priors ---
        out.create_dataset("upper_bound", data=np.array(model.upper))
        out.create_dataset("lower_bound", data=np.array(model.lower))

        # --- last position ---
        # FIXME: do this another way
        qlast = chain[-1, :]
        scene.set_all_source_params(qlast)
        patch.unzerocoords(scene)
        for i, source in enumerate(scene.sources):
            source.id = active[i]["source_index"]
        qcat = scene.to_catalog(extra_cols=["source_index"])
        qcat["source_index"][:] = active["source_index"]

    return qcat, mass_matrix


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
