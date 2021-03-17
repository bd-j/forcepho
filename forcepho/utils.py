# -*- coding: utf-8 -*-

import os, time
from argparse import Namespace
import numpy as np

from astropy.io import fits
import h5py

from .sources import Galaxy
from .reconstruction import _make_imset


__all__ = ["Logger",
           "read_config", "update_config",
           "sourcecat_dtype", "rectify_catalog",
           "extract_block_diag",
           "make_statscat", "make_chaincat",
           "write_residuals", "sky_to_pix"]


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


def read_config(config_file, args=None):
    """Read a yaml formatted config file.
    """
    import yaml
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config = Namespace()
    for k, v in config_dict.items():
        if type(v) is list:
            v = np.array(v)
        if "dtype" in k:
            v = np.typeDict[v]
        setattr(config, k, v)

    config = update_config(config, args)

    return config


def update_config(config, args):
    """Update a configuration namespace with parsed command line arguments.
    Also prepends config.store_directory to *storefile names
    """
    if args is not None:
        d = vars(args)
        for k, v in d.items():
            try:
                if v:
                    setattr(config, k, v)
            except:
                print("could not update {}={}".format(k, v))

    # update the store paths
    for store in ["pixel", "meta", "psf"]:
        try:
            attr = "{}storefile".format(store)
            n = getattr(config, attr)
            new = os.path.join(config.store_directory, n)
            setattr(config, attr, new)
        except(AttributeError):
            print("could not update {}storefile path".format(store))

    # expand shell variables
    for k, v in vars(config).items():
        try:
            s = os.path.expandvars(v)
            setattr(config, k, s)
        except(TypeError):
            pass

    return config


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
    colnames = [b.decode("utf") for b in bands] + [s.decode("utf") for s in shapes]
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


def write_residuals(model, filename):
    pixattrs = ["data", "xpix", "ypix", "ierr"]
    metas = ["D", "CW", "crpix", "crval"]
    patcher = model.proposer.patch
    epaths = patcher.epaths

    with h5py.File(filename, "w") as out:
        out.create_dataset("epaths", data=np.array(epaths, dtype="S"))
        out.create_dataset("exposure_start", data=patcher.exposure_start)
        out.create_dataset("reference_coordinates", data=patcher.patch_reference_coordinates)
        #out.create_dataset("active", data=)
        #out.create_dataset("fixed", data=fixed)

        for band in patcher.bandlist:
            g = out.create_group(band)

        #residual = np.split(model._residuals, np.cumsum(patcher.exposure_N)[:-1])
        _make_imset(out, epaths, "residual", model._residuals)

        for a in pixattrs:
            arr = patcher.split_pix(a)
            _make_imset(out, epaths, a, arr)

        for a in metas:
            arr = getattr(patcher, a)
            _make_imset(out, epaths, a, arr)


def sky_to_pix(self, ra, dec, exp=None, ref_coords=0.):
    """
    Parameters
    ----------
    ra : float (degrees)

    dec : float (degrees)

    exp : dict-like
        Must have the keys `crpix`, `crval`, and `CW` encoding the astrometry

    ref_coords : ndarray of shape (2,)
        The reference coordinates (ra, dec) for the supplied astrometry
    """
    # honestly this should query the full WCS using
    # get_local_linear for each ra,dec pair
    crval = exp["crval"][:]
    crpix = exp["crpix"][:]
    CW = exp["CW"][:]

    i = 0
    if len(CW) != len(ra):
        CW = CW[i]
        crval = crval[i]
        crpix = crpix[i]

    sky = np.array([ra, dec]).T - (crval + ref_coords)
    pix = np.matmul(CW, sky[:, :, None])[..., 0] + crpix

    return pix

