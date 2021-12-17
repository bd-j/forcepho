# -*- coding: utf-8 -*-

import os, time
from argparse import Namespace

import numpy as np
from scipy.special import gamma, gammainc, gammaincinv

import json
import h5py
from astropy.io import fits

from .sources import Galaxy


__all__ = ["Logger", "NumpyEncoder",
           "read_config", "update_config",
           "extract_block_diag",
           "make_statscat", "make_chaincat",
           "write_residuals", "make_imset", "sky_to_pix",
           "isophotal_radius", "kron_radius"]


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def read_config(config_file, args=None):
    """Read a yaml formatted config file.
    """
    import yaml
    if type(config_file) is str:
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
    elif type(config_file) is dict:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)

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
                if v is not None:
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


def write_residuals(patcher, filename, residuals=None):
    # TODO: Should this be a method on Patch or Posterior
    pixattrs = ["data", "xpix", "ypix", "ierr"]
    metas = ["D", "CW", "crpix", "crval"]
    epaths = patcher.epaths

    with h5py.File(filename, "w") as out:
        out.create_dataset("epaths", data=np.array(epaths, dtype="S"))
        #out.create_dataset("exposure_start", data=patcher.exposure_start)
        out.create_dataset("reference_coordinates", data=patcher.patch_reference_coordinates)
        #out.create_dataset("active", data=)
        #out.create_dataset("fixed", data=fixed)

        for band in patcher.bandlist:
            g = out.create_group(band)

        #residual = np.split(model._residuals, np.cumsum(patcher.exposure_N)[:-1])
        if residuals is not None:
            make_imset(out, epaths, "residual", residuals)

        for a in pixattrs:
            arr = patcher.split_pix(a)
            make_imset(out, epaths, a, arr)

        for a in metas:
            arr = getattr(patcher, a)
            make_imset(out, epaths, a, arr)


def make_imset(out, paths, name, arrs):
    for i, epath in enumerate(paths):
        try:
            g = out[epath]
        except(KeyError):
            g = out.create_group(epath)

        try:
            g.create_dataset(name, data=np.array(arrs[i]))
        except:
            print("Could not make {}/{} dataset from {}".format(epath, name, arrs[i]))


def sky_to_pix(ra, dec, exp=None, ref_coords=0.):
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


def frac_sersic(radius, rhalf=None, sersic=2.0):
    """For a given `rhalf` and `sersic` index, compute the fraction of flux
    falling within radius
    """
    from scipy.special import gamma, gammainc, gammaincinv
    g2n = gamma(2 * sersic)
    bn = gammaincinv(2*sersic, 0.5)  # note gammainc(a, x) is normalized by gamma(a)
    x = bn * (radius / rhalf)**(1/sersic)
    return gammainc(2*sersic, x)


def kron_radius(rhalf, sersic, rmax=None):
    k = gammaincinv(2 * sersic, 0.5)
    if rmax:
        x = k*(rmax / rhalf)**(1./sersic)
        c = gammainc(3 * sersic, x) / gammainc(2 * sersic, x)
    else:
        c = gamma(3 * sersic) / gamma(2 * sersic)
    r_kron = rhalf / k**sersic * c
    return r_kron


def I_eff(lum, rhalf, flux_radius=None, sersic=1):
    """gamma(2n, b_n) = 1/2; b_n = (re / r0)**(1/n)
    """
    two_n = 2 * sersic
    k = gammaincinv(two_n, 0.5)
    f =  gamma(two_n)
    if flux_radius is not None:
        x = k * (flux_radius / rhalf)**(1. / sersic)
        f *= gammainc(two_n, x)
    conv = k**(two_n) / (sersic * np.exp(k)) / f
    Ie = lum * conv / (2 * np.pi * rhalf**2)
    return Ie


def isophotal_radius(iso, flux, r_half, flux_radius=None, sersic=1):
    Ie = I_eff(flux, r_half, flux_radius=flux_radius, sersic=sersic)
    k = gammaincinv(2 * sersic, 0.5)
    r_iso = (1 - np.log(iso / Ie) / k)**(sersic)

    return r_iso * r_half

