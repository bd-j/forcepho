# -*- coding: utf-8 -*-

import os, argparse
from argparse import Namespace, ArgumentParser
import numpy as np


__all__ = ["main_config_parser", "preprocess_config_parser",
           "default_config_file",
           "parse_all", "read_config", "make_bounds_kwargs",
           "new_args", "yaml_to_namespace",
           "update_config", "rectify_config"]


def main_config_parser(FITS=True):

    parser = ArgumentParser()

    # -------
    # basic data types
    parser.add_argument("--pix_dtype", type=str, default="float32")
    parser.add_argument("--meta_dtype", type=str, default="float32")

    # -------
    # output
    parser.add_argument("--outbase", type=str, default="output/",
                        help=("Path to directory that will contain patch output "
                              "directories and state info"))
    parser.add_argument("--write_residuals", type=int, default=1,
                        help=("switch; turns on/off the writing of residual "
                              "pixel data from the last sample for each patch."))
    # -------
    # input peak & big-object lists
    parser.add_argument("--peak_catalog", type=str, default="",
                        help=("Path to the fpho formatted peak catalog giving the location "
                              "and initial parameter guesses of Sersic profiles to fit."))
    parser.add_argument("--big_catalog", type=str, default=None,
                        help="Deprecated")
    parser.add_argument("--bandlist", type=str, nargs="*", default=["F200W"])

    # -------
    # mixtures
    parser.add_argument("--splinedatafile", type=str, default="",
                        help="Path to Sersic mixture Gaussian amplitude lookup table.")
    parser.add_argument("--psfstorefile", type=str, default="",
                        help="Path to the file continaing the PSF approximations.")

    # -------
    # pixel data locations
    if FITS:
        # FITS filenames
        parser.add_argument("--fitsfiles", type=str, nargs="*", default=[],
                            help=("list of paths to the FITS image data to fit to. "
                            "Often overridden internally."))
    else:
        # data store
        parser.add_argument("--pixelstorefile", type=str, default=None)
        parser.add_argument("--metastorefile", type=str, default=None,
                            help="Path to the json file containing metadata for the images.")

    # -------
    # pixel data tweaks
    parser.add_argument("--max_snr", type=float, default=0,
                        help="Add a per-pixel extra model uncertainty to cap the S/N at this value.")
    parser.add_argument("--tweak_background", type=str, default=None,
                        help=("Name of keyword in header giving background values to subtract "
                              "from pixel data."))

    # -------
    # bounds
    parser.add_argument("--sqrtq_range", type=float, nargs=2, default=[0.4, 1],
                        help="sqrt(b/a)")
    parser.add_argument("--pa_range", type=float, nargs=2, default=[-2, 2],
                        help="radians")
    parser.add_argument("--rhalf_range", type=float, nargs=2, default=[0.03, 1.0],
                        help="arcsec")
    parser.add_argument("--sersic_range", type=float, nargs=2, default=[0.8, 6.0])
    parser.add_argument("--dpos", type=float, nargs=2, default=[0.06, 0.06],
                        help="arcsec")

    # -------
    # optimization
    parser.add_argument("--use_gradients", type=int, default=1,
                        help="switch; turns on/off the use of analytic gradients in optimization")
    parser.add_argument("--linear_optimize", type=int, default=0,
                        help=("switch; turns on/off the use of a linear flux optimization "
                              "conditional on the shapes determined during optimization."))
    parser.add_argument("--gtol", type=float, default=1e-5,
                        help="Argument for scipy optimizers.")
    parser.add_argument("--add_barriers", type=int, default=0,
                        help="switch; turn on/off the use of e^Beta barriers when optimizing.")

    # -------
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=256,
                        help="Number of posterior samples to compute.")
    parser.add_argument("--warmup", type=int, nargs="*", default=[256],
                        help="Number of iterations of warm-up/burn-in.")
    parser.add_argument("--max_treedepth", type=int, default=9,
                        help="log_2 of the max number of points along an HMC trajectory.")
    parser.add_argument("--full_cov", type=int, default=1,
                        help=("switch; tunr on/off the estimation of a full (as opposed to "
                        "diagonal) mass matrix during warm-up."))
    parser.add_argument("--progressbar", type=int, default=0,
                        help=("switch; turn on/off the printing of a tqdm sampling progress "
                              "bar."))

    # -------
    # dis-patching
    parser.add_argument("--target_niter", type=int, default=256)
    parser.add_argument("--patch_maxradius", type=float, default=15, help="arcsec")
    parser.add_argument("--patch_minradius", type=float, default=1, help="arcsec")
    parser.add_argument("--maxactive_per_patch", type=int, default=15)
    parser.add_argument("--strict", type=int, default=1, help="switch;")
    parser.add_argument("--max_active_fraction", type=float, default=0.1)
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


def parse_all(parser, argv):
    """Parse options coming from parser defaults, a configuration file, and the
    command line.  Order of presedence is:

    1. parser defaults
    2. yaml config file
    3. command line arguments *that are different than the parser default*
    """
    # default argument values
    defargs = parser.parse_args([])

    # arguments that were specified at the command line
    clargs = new_args(parser, argv[1:])

    # get name of config file, looking in clargs first
    config_file = getattr(clargs, "config_file", None)
    if config_file is None:
        config_file = getattr(defargs, "config_file", None)

    # arguments specified in yaml config file
    if config_file is not None:
        yargs = yaml_to_namespace(config_file)
    else:
        yargs = None

    # overwrite defaults with yaml values
    config = update_config(defargs, yargs)
    # overwrite defaults and yaml with CLI values
    config = update_config(config, clargs)
    # expand paths, get numpy dtype, turn lists to arrays
    config = rectify_config(config)

    return config


def make_bounds_kwargs(config):
    fields = [f"{f}_range" for f in ["sqrtq", "sersic", "rhalf", "pa"]]
    fields += ["dpos", "n_sig_flux", "n_pix", "pixscale"]
    bounds_kwargs = dict()
    for field in fields:
        val = getattr(config, field, None)
        if val is not None:
            bounds_kwargs[field] = val
    return bounds_kwargs


def yaml_to_namespace(config_file):
    """Read argument from a yaml file.  Turn into a Namespace

    Parameters
    ----------
    config_file : string or dict
        The configuration information, either as the name of a yaml file or as a
        dictionary of argument,value pairs

    Returns
    -------
    config : Namespace
    """
    import yaml
    if type(config_file) is str:
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
    elif type(config_file) is dict:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)

    config = Namespace(**config_dict)

    return config


def update_config(config, args, rectify=True):
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
    return config


def rectify_config(config):

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

    for k, v in vars(config).items():
        try:
            # expand shell variables
            s = os.path.expandvars(v)
            setattr(config, k, s)
        except(TypeError):
            pass
        # turn lists into arrays
        if type(v) is list:
            setattr(config, k, np.array(v))
        # turn dtype strings into dtypes
        if "dtype" in k:
            setattr(config, k, np.sctypeDict[v])

    config.bounds_kwargs = make_bounds_kwargs(config)

    return config


def new_args(parser, argv):
    nconf = Namespace()
    oconf = parser.parse_args(argv)
    for k, v in vars(oconf).items():
        #if v == parset.get_default(k):
        if f"--{k}" not in argv:  # Ugly HACK
            continue
        setattr(nconf, k, v)
    return nconf


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
    config = yaml_to_namespace(config_file)
    config = update_config(config, args)
    config = rectify_config(config)

    return config
