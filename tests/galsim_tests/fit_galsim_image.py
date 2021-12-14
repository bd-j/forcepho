#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

from astropy.io import fits

from forcepho.patches import SimplePatch
from forcepho.superscene import LinkedSuperScene, rectify_catalog
from forcepho.utils import NumpyEncoder, read_config, adjust_bounds
from child import accomplish_task, optimization, sampling

from make_galsim_image import make_galsim_image, get_grid_params

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


__all__ = ["get_superscene",
           "fit_test_image"]


def get_superscene(config, logger, **rectify_kwargs):

    # --- Patch Dispatcher ---  (parent)
    if type(config.raw_catalog) is str:
        logger.info(f"reading catalog from {config.raw_catalog}")
        try:
            unc = fits.getdata(config.raw_catalog, 2)
            config.bounds_kwargs.update(unccat=unc)
            logger.info(f"Flux priors based on spplied uncertainty estimates.")
        except(IndexError):
            pass

    cat, bands, chdr = rectify_catalog(config.raw_catalog, **rectify_kwargs)
    bands = [b for b in bands if b in config.bandlist]

    try:
        roi = cat["roi"]
        if roi.max() <= 0:
            roi = 2 * cat["rhalf"]
            logger.info("using twice rhalf for ROI")
    except(IndexError):
        logger.info("using twice rhalf for ROI")
        roi = 2 * cat["rhalf"]

    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               maxactive_per_patch=config.maxactive_per_patch,
                               maxradius=config.patch_maxradius,
                               minradius=getattr(config, "patch_minradius", 1.0),
                               target_niter=config.sampling_draws,
                               statefile=os.path.join(config.outbase, config.scene_catalog),
                               bounds_kwargs=config.bounds_kwargs,
                               strict=config.strict,
                               roi=roi)
    sceneDB = adjust_bounds(sceneDB, bands, config)
    logger.info("Made SceneDB")
    return sceneDB, bands


def fit_test_image(target_image, config, taskID=0, logger=None):
    """Fit a test image with forcepho
    """

    if config.optimize:
        method = optimization
    else:
        method = sampling

    # --- get the FITS image name and catalog ---
    config.fitsfiles = [target_image]
    assert len(config.fitsfiles) == 1
    n = config.fitsfiles[0]
    config.raw_catalog = fits.getdata(n, 2), fits.getheader(n, 2)

    # --- Get patch dispatcher and maker ---
    sceneDB, bands = get_superscene(config, logger, sqrtq_range=(0.35, 0.99))
    patcher = SimplePatch(psfstore=config.psfstorefile,
                          splinedata=config.splinedatafile,
                          fitsfiles=config.fitsfiles,
                          return_residual=True)

    # --- checkout scenes --- (parent)
    region, active, fixed = sceneDB.checkout_region(seed_index=-1)
    big = None
    logger.info(f"Checked out scene centered on source index="
                f"{active[0]['source_index']} with ID={active[0]['id']}")

    # build the chore
    bounds, cov = sceneDB.bounds_and_covs(active["source_index"])
    chore = {'region': region,
             'active': active, 'fixed': fixed, 'big': big,
             'bounds': bounds, 'cov': cov, 'taskID': taskID,
             'bands': bands, 'shape_cols': sceneDB.shape_cols}

    # submit the task and get the results
    if HASGPU:
        result = accomplish_task(patcher, chore, config, logger, method=method)
        logger.info(f"Got results for {chore['taskID']}.")

    else:
        patcher._dirty_data = False
        patcher.build_patch(region, allbands=bands)

        return patcher

    # Check results back in
    sceneDB.checkin_region(result['final'], result['out'].fixed,
                           config.sampling_draws,
                           block_covs=result['covs'],
                           taskID=taskID)
    sceneDB.writeout()
    logger.info(f"writing to {os.path.dirname(sceneDB.statefilename)}.")
    logger.info(f'SuperScene is done, shutting down.')

    return patcher


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--config_file", type=str, default="./test_config.yml")
    parser.add_argument("--psfstorefile", type=str, default=None)
    parser.add_argument("--test_grid", type=str, default="./galsim_grid.fits")
    parser.add_argument("--grid_index_start", type=int, default=0)
    parser.add_argument("--n_grid", type=int, default=10)
    parser.add_argument("--target_image", type=str, default="")
    # bounds
    parser.add_argument("--minflux", type=float, default=None)
    parser.add_argument("--maxfluxfactor", type=float, default=0)
    # output
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--outbase", type=str, default="../output/test")
    # optimization
    parser.add_argument("--optimize", type=int, default=0,
                        help="switch to optimize instead of sampling")
    parser.add_argument("--use_gradients", type=int, default=1)
    parser.add_argument("--linear_optimize", type=int, default=0)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--add_barriers", type=int, default=0)
    # sampling
    parser.add_argument("--full_cov", type=int, default=0)
    parser.add_argument("--sampling_draws", type=int, default=None)
    parser.add_argument("--max_treedepth", type=int, default=None)
    parser.add_argument("--warmup", type=int, nargs="*", default=None)
    parser.add_argument("--progressbar", type=int, default=0)
    parser.add_argument("--discard_tuning", type=int, default=1)

    # --- Logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('dispatcher-parent')
    #logger.info(f'Starting parent on {socket.gethostname()}')

    # --- Configure ---
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)
    if os.path.exists(config.test_grid):
        _ = shutil.copy(config.test_grid, config.outbase)
    try:
        with open(f"{config.outbase}/config.json", "w") as cfg:
            json.dump(vars(config), cfg, cls=NumpyEncoder)
    except(ValueError):
        logger.info("Config json not written.")

    # --- Decide what the tasks are ---
    if not config.target_image:
        start = config.grid_index_start
        tasklist = list(range(s, s + config.n_grid))
    else:
        tasklist = [config.target_image]

    # --- loop over tasks ---
    targets = []
    for taskID in tasklist:
        logger.info(f"working on taskID = {taskID}")

        if type(taskID) is int:
            #  --- get/make the test image? ---
            logger.info("making test image")

            params = get_grid_params(config, taskID=taskID)
            parstring = "band={band}, rhalf={rhalf}, sersic={sersic}, q={q}, snr={snr}".format(**params)
            logger.info(f"parameters are {parstring}")

            test_image_name = make_test(params["band"], params["rhalf"],
                                        params["sersic"], params["snr"], params["q"],
                                        flux=1, nx=196, ny=196,  # note size to ensure region within image
                                        psf_type="webbpsf", psffile=config.psfstorefile,
                                        outname=None)
            logger.info(f"wrote galsim image to {test_image_name}")
            targets.append(test_image_name)
            config.bandlist = [params["band"]]
        else:
            # --- use a supplied test image name ---
            test_image_name = taskID
            taskID = 0

        # --- fit the test image ---
        config.target_image = test_image_name
        if os.path.exists(config.target_image):
            patcher = fit_test_image(config, logger=logger, taskID=taskID)
        else:
            logger.error(f"{config.target_image} does not exist.")