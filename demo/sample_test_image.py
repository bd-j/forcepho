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


def bounds(active, filternames, ref=np.array([0., 0.]), dra=None, ddec=None,
           rhalf_range=(0.03, 0.3), sersic_range=(1., 5.)):

    npix, pixscale = 2, 0.03
    if dra is None:
        dra = npix * pixscale / 3600. / np.cos(np.deg2rad(-27))
    if ddec is None:
        ddec = npix * pixscale / 3600.

    shape_lower = [0.3, -0.6 * np.pi, sersic_range[0], rhalf_range[0]]
    shape_upper = [1.0,  0.6 * np.pi, sersic_range[1], rhalf_range[1]]

    lower, upper = [], []
    for row in active:
        fluxes = np.array([row[f] for f in filternames])
        sigma_flux = 0.5 * np.abs(fluxes)
        flux_lower = fluxes - sigma_flux
        flux_upper = fluxes + sigma_flux
        pos_lower = [row["ra"]-ref[0]-dra, row["dec"]-ref[1]-ddec]
        pos_upper = [row["ra"]-ref[0]+dra, row["dec"]-ref[1]+ddec]
        lower.extend(flux_lower.tolist() + pos_lower + shape_lower)
        upper.extend(flux_upper.tolist() + pos_upper + shape_upper)

    return np.array(lower), np.array(upper)


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
    logger.info("Made SceneDB")

    # --- Sample the patches ---
    while sceneDB.undone:
        region, active, fixed = sceneDB.checkout_region()
        if active is None:
            continue
        logger.info("Checked out scene with {} active sources".format(len(active)))

        # --- Build patch and fit ---
        patchID = int(active["source_index"][0])
        patcher.build_patch(region, None, allbands=bands)
        proposer, q = prepare(patcher, active=active, fixed=fixed)
        logger.info("Prepared Patch {:04.0f}".format(patchID))

        lower, upper = bounds(active, patcher.bandlist, ref=patcher.patch_reference_coordinates)
        if np.any(q < lower) or np.any(q > upper):
            logger.info("starting position out of bounds")
            print(q)
            raise ValueError("starting position out of bounds")
        model = make_model(proposer, lower, upper)
        logger.info("Model made, sampling with {} warmup and {} draws")
        out, step, stats = run_lmc(model, q.copy(), config.sampling_draws,
                                    warmup=config.warmup, progressbar=True)

        # --- Deal with results ---
        logger.info("Sampling complete, preparing output.")
        out, final, mass_matrix = make_result(out, region, active, fixed, model,
                                              step=step, stats=stats, patchID=patchID)
        logger.info("Checking region back in")
        sceneDB.checkin_region(final, fixed, config.sampling_draws, mass_matrix=mass_matrix)

        outfile = os.path.join(args.patch_dir, "patch{:04.0f}_results.h5".format(patchID))
        logger.info("Writing to {}".format(outfile))
        out.dump_to_h5(outfile)

    sceneDB.writeout()