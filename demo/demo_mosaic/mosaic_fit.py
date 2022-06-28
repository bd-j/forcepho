#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
from forcepho.superscene import LinkedSuperScene
from forcepho.utils import NumpyEncoder, write_to_disk
from forcepho.fitting import run_lmc

from demo_utils import write_fits_to


class Patcher(FITSPatch, CPUPatchMixin):
    pass


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--psfstorefile", type=str, default="./single_gauss_psf.h5")
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--image_basename", type=str, default="./data/dither")
    # output
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="./output/")
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=2048)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=1)

    # --- Logger ---
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('demonstrator-mosaic')

    # --- Configure ---
    config = parser.parse_args()
    [os.makedirs(a, exist_ok=True) for a in (config.outdir,)]
    with open(os.path.join(config.outdir, "config.json"), "w") as cfg:
        json.dump(vars(config), cfg, cls=NumpyEncoder)
    logger.info(f"Configured, writing to {config.outdir}.")

    config.image_names = glob.glob(config.image_basename + "*.fits")
    outname = os.path.basename(config.image_names[0]).split("_")[0]
    logger.info(f"Got {len(config.image_names)} image matching {config.image_basename}.")
    taskID = 0

    # --- build the scene server ---
    cat = fits.getdata(config.image_names[0], -1)
    bands = fits.getheader(config.image_names[0], -1)["FILTERS"].split(",")
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "superscene.fits"),
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=dict(n_pix=1.0),
                               target_niter=config.sampling_draws)
    logger.info("Made SceneDB")

    # --- load the image data ---
    patcher = Patcher(fitsfiles=config.image_names,
                      psfstore=config.psfstorefile,
                      splinedata=config.splinedatafile,
                      return_residual=True)

    # --- check out scene  & bounds ---
    region, active, fixed = sceneDB.checkout_region(seed_index=-1)
    bounds, cov = sceneDB.bounds_and_covs(active["source_index"])
    logger.info(f"Checked out scene and bounds.")

    # --- prepare model and data, and sample ---
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=sceneDB.shape_cols)
    out, step, stats = run_lmc(model, q.copy(),
                               n_draws=config.sampling_draws,
                               warmup=config.warmup,
                               z_cov=cov, full=True,
                               weight=max(10, active["n_iter"].min()),
                               discard_tuned_samples=False,
                               max_treedepth=config.max_treedepth,
                               progressbar=config.progressbar)

    # --- Check results back in and end ---
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=taskID)
    outroot = os.path.join(config.outdir, os.path.basename(config.image_basename))
    logger.info(f"Writing to {outroot}")
    write_to_disk(out, outroot, model, config)
    sceneDB.checkin_region(final, fixed,
                           config.sampling_draws,
                           block_covs=covs,
                           taskID=taskID)
    sceneDB.writeout()
    logger.info(f'SuperScene is done, shutting down.')
