#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sample_test_image.py - Fit a scene to a test image, using littlemcmc
"""

import os, sys, argparse, time
import numpy as np
from astropy.io import fits
# import h5py

from forcepho.dispatcher import SuperScene
from forcepho.sources import Galaxy
from forcepho.patches import JadesPatch

from forcepho.fitting import Result, run_lmc
from forcepho.utils import Logger, rectify_catalog, read_config

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def do_child(patcher, task, config):
    rank = 1
    global logger
    logger = logging.getLogger(f'dispatcher-child-{rank}')

    # --- Event Loop ---
    for taskid in range(1):
        logger.info(f'Received task')
        # if shutdown break and quit
        if task is None:
            break

        # --- unpack all the task variables ---
        region, active, fixed = task['region'], task['active'], task['fixed']
        bounds, cov = task['bounds'], task['cov']
        bands, shape_cols = task['bands'], task['shape_cols']
        sourceID = active[0]["source_index"]
        del task
        logger.info(f"Child {rank} received RA {region.ra}, DEC {region.dec} with tag {taskid}")
        logger.info(f"source ID: {sourceID}")

        # --- Build patch & prepare model--- (child)
        patcher.build_patch(region, None, allbands=bands)
        model, q = patcher.prepare_model(active=active, fixed=fixed,
                                         bounds=bounds, shapes=sceneDB.shape_cols)
        logger.info("Prepared patch and model")
        if config.sampling_draws == 0:
            return patcher

        # --- Sample using covariances--- (child)
        weight = max(10, active["n_iter"].min())
        logger.info(f"sampling with covariance weight={weight}")
        out, step, stats = run_lmc(model, q.copy(), config.sampling_draws,
                                   full=config.full_cov, z_cov=cov, adapt=True,
                                   weight=weight, warmup=config.warmup,
                                   progressbar=getattr(config, "progressbar", False))
        logger.info(f"Sampling complete ({model.ncall} calls), preparing output.")

        # --- develop the payload ---
        final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                               step=step, stats=stats, patchID=taskid)
        payload = dict(out=out, final=final, covs=covs)

        # --- write the output for this task ---
        outfile = os.path.join(config.patch_dir, f"patch{taskid}_samples.h5")
        logger.info(f"Writing to {outfile}")
        #out.config = json.dumps(vars(config))
        out.dump_to_h5(outfile)

        # --- Write image data and residuals if requested ---
        if config.write_residuals:
            outfile = os.path.join(config.patch_dir, f"patch{taskid}_residuals.h5")
            logger.info(f"Writing residuals to {outfile}")
            patcher.return_residual = True
            z = out.chain[-1, :]  # last position in chain
            model.evaluate(z)
            write_residuals(model, outfile)
            patcher.return_residual = False

        # --- send to parent, free GPU memory ---
        logger.info(f"Child {rank} sent {region.ra} for patch {taskid}")

    return payload


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./galsim_config.yml")
    parser.add_argument("--outbase", type=str, default="./output/sample_galsim/")
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--full_cov", type=int, default=1)
    parser.add_argument("--write_residuals", type=int, default=1)
    parser.add_argument("--progressbar", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)

    logger = Logger(__name__)

    # --- Wire the data --- (child)
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)
    logger.info("Data loaded, HASGPU={}".format(HASGPU))

    # --- Get the patch dispatcher ---  (parent)
    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    sceneDB = SuperScene(sourcecat=cat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch,
                         maxradius=config.patch_maxradius,
                         target_niter=config.sampling_draws,
                         statefile=os.path.join(args.outbase, args.scene_catalog),
                         bounds_kwargs={})
    logger.info("Made SceneDB")

    # --- Sample the patches ---
    while sceneDB.undone:
        # --- checkout a scene --- (parent)
        for _ in range(100):
            region, active, fixed = sceneDB.checkout_region()
            if active is not None:
                break
        bounds, cov = sceneDB.bounds_and_covs(active["source_index"])

        # --- send to child ---
        chore = {'region': region, 'active': active, 'fixed': fixed,
                 'bounds': bounds, 'cov': cov,
                 'bands': bands, 'shape_cols': sceneDB.shape_cols}
        result = do_child(patcher, chore, config)

        # --- deal with results ---
        logger.info("Checking region back in")
        sceneDB.checkin_region(result['final'], result['out'].fixed,
                               len(result['out'].chain),
                               block_covs=result['covs'],
                               taskID=1)

    sceneDB.writeout()