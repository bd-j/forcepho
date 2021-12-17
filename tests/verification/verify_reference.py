#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil, time
import argparse, logging, json
import numpy as np

from astropy.io import fits

import forcepho
from forcepho.patches import SimplePatch
from forcepho.utils import NumpyEncoder, read_config, write_residuals
from forcepho.sources import Galaxy

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./verification_config.yml")
    parser.add_argument("--reference_image", type=str, default="")
    parser.add_argument("--bandlist", type=str, nargs="*", default=None)

    # --- Configure ---
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)

    # --- Logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('verifier')

    # --- get the reference data, image and catalog ---
    config.fitsfiles = [config.reference_image]
    n = config.fitsfiles[0]
    with fits.open(n) as hdul:
        active, hdr = np.array(hdul[-1].data), hdul[-1].header
    bands = hdr["FILTERS"].split(",")
    bands = [b.upper() for b in bands]
    logger.info("Configured")

    # --- Build the patch with pixel data ---
    patcher = SimplePatch(psfstore=config.psfstorefile,
                          splinedata=config.splinedatafile,
                          fitsfiles=config.fitsfiles,
                          return_residual=True)
    logger.info("Instantiated patcher.")
    patcher.build_patch(region=None, allbands=bands)
    logger.info("Built patch.")

    if not HASGPU:
        logger.info("No GPU avaialable for further tests.")
        sys.exit()

    # --- Build the model with scene ---
    shape_cols = Galaxy.SHAPE_COLS
    model, q = patcher.prepare_model(active=active, shapes=shape_cols)
    logger.info("Prepped model.")

    # CHECK
    # --- evaluate the model ---
    residuals = model.residuals(q)
    model_image = patcher.data - residuals
    logger.info("Computed residuals.")

    # --- write the results ---
    vers = forcepho.__version__
    ts = time.strftime("%Y%b%d", time.localtime())
    out = f"output/verification_residuals_V{vers}.h5"
    write_residuals(patcher, out, residuals=[residuals])
    logger.info(f"Wrote residuals to {out}")

    # check valid pixels
    valid = patcher.xpix >= 0
    if not np.allclose(residuals[0, valid], 0, atol=5e-7):
        diff = np.nanmax(np.abs(residuals[0, valid]))
        raise ValueError(f"Model did not match reference image: max abs diff = {diff}")
    else:
        logger.info("Model matches reference, yay!")


    # --- lnp, lnp_grad values ---
    #z = model.transform.inverse_transform(q).copy()
    #model.evaluate(z)