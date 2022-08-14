# -*- coding: utf-8 -*-

"""chain.py - utilities for working with structured chain catalog arrays
"""

import numpy as np
from ..sources import Galaxy


__all__ = ["make_statscat", "make_chaincat", "get_sample_cat"]


def make_statscat(stats, step):
    """Convert the `stats` dictionary returned by littlemcmc to a structured
    array.  This should probably go in fitting.py
    """
    # Reshape `stats` to an array
    dtype = np.dtype(list(step.stats_dtypes[0].items()))
    stats_arr = np.zeros(len(stats), dtype=dtype)
    for c in stats_arr.dtype.names:
        stats_arr[c][:] = np.array([s[c] for s in stats])
    return stats_arr


def make_chaincat(chain, bands, active, ref, shapes=Galaxy.SHAPE_COLS):
    """
    """
    # --- Get sizes of things ----
    n_iter, n_param = chain.shape
    n_band = len(bands)

    n_param_per_source = n_band + len(shapes)
    assert (np.mod(n_param, n_param_per_source) == 0)
    n_source = int(n_param / n_param_per_source)
    assert (n_source == len(active))

    # --- generate dtype ---
    try:
        colnames = [b.decode("utf") for b in bands]
    except(AttributeError):
        colnames = list(bands)
    try:
        colnames += [s.decode("utf") for s in shapes]
    except(AttributeError):
        colnames += list(shapes)
    cols = [("source_index", np.int32)] + [(c, np.float64, (n_iter,))
                                           for c in colnames]
    dtype = np.dtype(cols)

    # --- make and fill catalog
    cat = np.zeros(n_source, dtype=dtype)
    cat["source_index"][:] = active["source_index"]
    for s in range(n_source):
        for j, col in enumerate(colnames):
            cat[s][col] = chain[:, s * n_param_per_source + j]

    # -- rectify parameters ---
    cat["ra"] += ref[0]
    cat["dec"] += ref[1]

    return cat


def combine_chains(chaincat, bands, groups={}):
    """Take a chain cat and a dictionary specifieying sources to be combined,
    and combine the fluxes for the chains.
    """
    raise NotImplementedError

    hdr = {}
    scat = chaincat
    ocat = np.zeros(len(groups), dtype=scat.dtype)

    mid = scat["id"].tolist()
    for i, (g, sub) in enumerate(groups.items()):
        ocat[i]["id"] = float(g)
        hdr[f"G{g}"] = ",".join(sub)
        ids = [float(f"{g}{s}") for s in sub]
        inds = [mid.index(i) for i in ids]
        for b in bands:
            chain = scat[inds][b].sum(axis=0)
            ocat[i][b] = chain
        for c in ["ra", "dec"]:
            # straight mean, nto a barycenter
            chain = scat[inds][c].mean(axis=0)
            ocat[i][c] = chain

    return ocat, hdr


def get_sample_cat(chaincat, iteration, active):
    """Get a sample of the scene parameters from the chain, as a structured
    arrays with one row for each source.

    Parameters
    ----------
    chaincat : structured ndarray
        The chain catalog

    iteration : int
        The iteration of the chain for which to produce a catalog.

    active : structured ndarray
        The ndarray representing the active scene

    Returns
    -------
    sample : structured ndarray of shape (n_active,)
        The parameters of each source at the specified iteration of the
        chain, as a structured array.
    """
    #dtype_sample = np.dtype([desc[:2] for desc in self.chaincat.dtype.descr])
    #sample = np.zeros(self.n_active, dtype=dtype_sample)
    sample = active.copy()
    for d in sample.dtype.names:
        if d in chaincat.dtype.names:
            try:
                sample[d] = chaincat[d][:, iteration]
            except(IndexError):
                sample[d] = chaincat[d]
    return sample