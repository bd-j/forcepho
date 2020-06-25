#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import argparse
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.dispatcher import SuperScene
from forcepho.sources import Galaxy
from forcepho.proposal import Proposer
from forcepho.patches import JadesPatch

from forcepho.model import GPUPosterior, BoundedTransform
from forcepho.fitting import run_lmc

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
    if HASGPU:
        gpu_patch = patcher.send_to_gpu()
    proposer = Proposer(patcher)
    proposer.patch.return_residual = True

    # --- Get parameter vector and proposal
    q = patcher.scene.get_all_source_params().copy()

    # --- subtract fixed sources ---
    if (fixed is not None) & HASGPU:
        q_fixed = q
        prop_fixed = patcher.scene.get_proposal()
        out = proposer.evaluate_proposal(prop_fixed)
        residual_fixed = out[-1]

        patcher.pack_meta(active)
        q = patcher.scene.get_all_source_params().copy()
        patcher.swap_on_gpu()

    proposer.patch.return_residual = False

    return proposer, q


def bounds(scene, active=None):
    return lower, upper


def make_model(proposer, q, **model_kwargs):
    """Make a model including prior bounds
    """
    ndim = len(q)
    lower, upper = bounds(proposer.patcher.scene)

    transform = BoundedTransform(lower, upper)
    model = GPUPosterior(proposer, proposer.patcher.scene,
                         transform=transform,
                         **model_kwargs)

    return model, q


if __name__ == "__main__":

    # --- Wire the data ---
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)

    # --- Get the patch dispatcher ---
    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    sceneDB = SuperScene(sourcecat=cat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch,
                         maxradius=config.patch_maxradius,
                         target_niter=config.sampling_draws)
    sceneDB.parameter_columns = Galaxy.SHAPE_COLS + bands

    # --- Sample the patches ---
    while sceneDB.undone:
        region, active, fixed = sceneDB.checkout_region()
        if active is None:
            continue

        # --- Build patch and fit ---
        patcher.build_patch(region, None, allbands=bands)
        proposer, q = prepare(patcher, active=active, fixed=fixed)
        model = make_model(proposer, q.copy())
        chain, step, stats = run_lmc(model, q.copy(), config.sampling_draws,
                                     warmup=config.warmup)

        # --- Deal with results ---
        out, final, mass_matrix = make_result(active, fixed, region, model, chain,
                                              step=step, stats=stats, start=q.copy())
        sceneDB.checkin_region(qcat, final, draws, mass_matrix=mass_matrix)
        out.dump_to_h5()