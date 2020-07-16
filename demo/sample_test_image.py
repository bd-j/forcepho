#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, argparse, time
import numpy as np

from forcepho.dispatcher import SuperScene
from forcepho.sources import Galaxy
from forcepho.proposal import Proposer
from forcepho.patches import JadesPatch

from forcepho.model import GPUPosterior, BoundedTransform
from forcepho.fitting import Result, run_lmc

from utils import Logger, rectify_catalog, make_result
from utils import make_bounds, bounds_vectors
from config_test import config

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def prepare(patcher, active=None, fixed=None):
    """Prepare the patch for model making
    """
    if fixed is not None:
        cat = fixed
    else:
        cat = active

    # --- Build the things
    patcher.pack_meta(cat)
    patcher.return_residual = True
    gpu_patch = patcher.send_to_gpu()
    proposer = Proposer(patcher)

    # --- Get parameter vector and proposal
    q = patcher.scene.get_all_source_params().copy()

    # --- subtract fixed sources ---
    if (fixed is not None):
        q_fixed = q
        prop_fixed = patcher.scene.get_proposal()
        out = proposer.evaluate_proposal(prop_fixed)
        residual_fixed = out[-1]

        patcher.pack_meta(active)
        q = patcher.scene.get_all_source_params().copy()
        patcher.swap_on_gpu()

    proposer.patch.return_residual = False

    return proposer, q


def make_model(proposer, lower, upper, **model_kwargs):
    """Make a model including prior bounds
    """
    transform = BoundedTransform(lower, upper)
    model = GPUPosterior(proposer, proposer.patch.scene,
                         transform=transform,
                         **model_kwargs)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--patch_dir", type=str, default="../output/")
    args = parser.parse_args()

    if args.logging:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
    else:
        logger = Logger(__name__)

    # --- Wire the data ---
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)
    logger.info("Data loaded, HASGPU={}".format(HASGPU))

    # --- Get the patch dispatcher ---
    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    sceneDB = SuperScene(sourcecat=cat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch,
                         maxradius=config.patch_maxradius,
                         target_niter=config.sampling_draws)
    sceneDB.bounds_catalog = make_bounds(sceneDB.sourcecat, bands)
    logger.info("Made SceneDB")

    # --- Sample the patches ---
    while sceneDB.undone:
        # --- checkout a scene ---
        region, active, fixed = sceneDB.checkout_region()
        bounds = sceneDB.bounds_catalog[active["source_index"]]
        if active is None:
            continue
        logger.info("Checked out scene with {} active sources".format(len(active)))

        # --- Build patch ---
        patchID = int(active["source_index"][0])
        patcher.build_patch(region, None, allbands=bands)
        proposer, q = prepare(patcher, active=active, fixed=fixed)
        logger.info("Prepared Patch {:04.0f}".format(patchID))

        # --- Get bounds and sample ---
        lower, upper = bounds_vectors(bounds, patcher.bandlist,
                                      reference_coordinates=patcher.patch_reference_coordinates)
        model = make_model(proposer, lower, upper)
        logger.info("Model made, sampling with {} warmup and {} draws")
        out, step, stats = run_lmc(model, q.copy(), config.sampling_draws,
                                    warmup=config.warmup, progressbar=True)

        # --- Deal with results ---
        logger.info("Sampling complete, preparing output.")
        out, final, mass_matrix = make_result(out, region, active, fixed, model, bounds=bounds,
                                              step=step, stats=stats, patchID=patchID)
        logger.info("Checking region back in")
        sceneDB.checkin_region(final, fixed, config.sampling_draws, mass_matrix=mass_matrix)

        outfile = os.path.join(args.patch_dir, "patch{:04.0f}_results.h5".format(patchID))
        logger.info("Writing to {}".format(outfile))
        out.dump_to_h5(outfile)

    sceneDB.writeout()