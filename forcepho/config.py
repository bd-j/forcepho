# -*- coding: utf-8 -*-

import argparse
from argparse import Namespace, ArgumentParser
import numpy as np


__all__ = ["config_parser",
           "read_config", "update_config"]


def main_config_parser(FITS=True):

    parser = ArgumentParser()

    # -------
    # output
    parser.add_argument("--outbase", type=str, default="output/",
                        help="Path to directory that will contain patch output directories and state info")
    parser.add_argument("--write_residuals", type=int, default=1, help="switch;")

    # -------
    # input peak & big-object lists
    parser.add_argument("--peak_catalog", type=str, default="")
    parser.add_argument("--big_catalog", type=str, default=None)
    parser.add_argument("--bandlist", type=str, nargs="*", default=["F200W"])

    # -------
    # mixtures
    parser.add_argument("--splinedatafile", type=str, default="")
    parser.add_argument("--psfstorefile", type=str, default="")

    # -------
    # pixel data locations
    if FITS:
        # FITS filenames
        parser.add_argument("--fitsfiles", type=str, nargs="*", default=[])
    else:
        # data store
        parser.add_argument("--pixelstorefile", type=str, default=None)
        parser.add_argument("--metastorefile", type=str, default=None)

    # -------
    # pixel data tweaks
    parser.add_argument("--max_snr", type=float, default=0)
    parser.add_argument("--tweak_background", type=str, default=None,
                        help="Name of keyword in header giving background values to subtract from pixel data")

    # -------
    # basic data types
    parser.add_argument("--pix_dtype", type=str, default="float32")
    parser.add_argument("--meta_dtype", type=str, default="float32")

    # -------
    # bounds
    parser.add_argument("--sqrtq_range", type=float, nargs=2, default=[0.4, 1], help="sqrt(b/a)")
    parser.add_argument("--pa_range", type=float, nargs=2, default=[-2, 2], help="radians")
    parser.add_argument("--rhalf_range", type=float, default=[0.03, 1.0], help="arcsec")
    #parser.add_argument("--delta_pos", type=float, nargs=2, default=[0.06, 0.06], help="arcsec")
    parser.add_argument("--pixscale", type=float, default=0.03, help="arcsec per pixel")
    parser.add_argument("--npix", type=float, default=2.0, help="width for positional pbounds in pixels")

    # -------
    # optimization
    parser.add_argument("--use_gradients", type=int, default=1, help="switch;")
    parser.add_argument("--linear_optimize", type=int, default=0, help="switch;")
    parser.add_argument("--gtol", type=int, default=0.00001)
    parser.add_argument("--add_barriers", type=int, default=0, help="switch;")

    # -------
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=256)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--max_treedepth", type=int, default=9)
    parser.add_argument("--full_cov", type=int, default=1, help="switch;")
    parser.add_argument("--progressbar", type=int, default=0, help="switch;")

    # -------
    # dis-patching
    parser.add_argument("--target_niter", type=int, default=256)
    parser.add_argument("--patch_maxradius", type=float, default=15, help="arcsec")
    parser.add_argument("--maxactive_per_patch", type=int, default=15)
    parser.add_argument("--strict", type=int, default=1, help="switch;")
    parser.add_argument("--max_active_fraction", typefloat, default=0.1)
    parser.add_argument("--ntry_checkout", type=int, default=1000)
    parser.add_argument("--buffer_size", type=float, default=5e7)

    return parser


def preprocess_config_parser():
    parser = ArgumentParser()

    # image store creation
    parser.add_argument("--do_fluxcal", type=int, default=1, help="switch;")
    parser.add_argument("--bitmask", type=int, default=1)
    parser.add_argument("--super_pixel_size", type=int, default=8)
    parser.add_argument("--nside_full", type=int, nargs=2, default=[2048, 2048])

    # data store
    parser.add_argument("--pixelstorefile", type=str, default=None)
    parser.add_argument("--metastorefile", type=str, default=None)

    return parser


def default_config_file(filename="", **kwargs):
    parser = main_config_parser(**kwargs)
    config = parser.parse_args()
    import yaml
    configtext = yaml.dumps(vars(config))
    if filename:
        with open(filename, "w") as cfile:
            cfile.write(ctext)
    return cfile


def read_config(config_file, args=None):
    """Read a yaml formatted config file into a Namespace.  This also expands
    shell variables in any configuration parameters, and prepends the value of
    `store_directory` to the `psfstorefile`, `pixelstorefile` and
    `metastorefile`

    Parameters
    ----------
    config_file : string
        Path to yaml formatted file with configuration parameters

    args : Namespace, optional
        Namespace of additional arguments that will override settings in the
        yaml file.

    Returns
    -------
    config : `argparse.Namespace()` instance.
        The configuration parameters as attributes of a Namespace.
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
    """Update a configuration namespace with parsed command line arguments. Also
    prepends config.store_directory to *storefile names, and expands shell
    variables.
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
    if getattr(config, "store_directory", ""):
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
