# -*- coding: utf-8 -*-

""" tasks.py - Thin wrappers on optimization and sampling methods to allow more
flexibility when accomplishing tasks in a parallelized mode.
"""
import argparse
import numpy as np

from .fitting import run_lmc, run_opt, optimize_fluxes
from .utils import make_chaincat, get_sample_cat
from .superscene import flux_bounds


def optimize(patcher, scene, config, logger,
             scratchpad=argparse.Namespace()):

    # --- unpack task ---
    region, active, fixed, bounds = scene

    # --- prepare data and model ---
    patcher.build_patch(region, None, allbands=config.bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds)
    scratchpad.q_initial = q.copy()
    cov = np.eye(len(q))

    # --- get a new start position randomly drawn from prior ---
    q = np.random.uniform(model.transform.lower, model.transform.upper)
    scratchpad.q_start = q.copy()
    scratchpad.active_start = active.copy()
    scratchpad.bounds_start = bounds.copy()

    # --- Add steep penalty near prior edges? ---
    if config.add_barriers:
        logger.info("Adding Barriers")
        from forcepho.priors import ExpBeta
        model._lnpriorfn = ExpBeta(model.transform.lower, model.transform.upper)

    # --- Do BFGS optimization ---
    logger.info("BFGS optimization")
    model.sampling = False
    opt, scires = run_opt(model, q.copy(), jac=True, gtol=config.gtol)
    out, step, stats = opt, None, scires
    model.sampling = True
    if config.add_barriers:
        model._lnpriorfn = None
        logger.info("Removed Barriers")

    # --- get new starting position and catalog ---
    q = opt.chain[-1]
    postop_chain = make_chaincat(opt.chain, patcher.bandlist, active,
                                 patcher.patch_reference_coordinates)
    active = get_sample_cat(postop_chain, -1, active)
    #model.scene.set_all_source_params(q_start)
    #active = model.scene.to_catalog()
    scratchpad.q_postop = q.copy()
    scratchpad.active_postop = active.copy()
    scratchpad.bounds_postop = bounds.copy()

    # --- Do linear flux optimization?
    if config.linear_optimize:
        logger.info("Beginning linear optimization")
        result = optimize_fluxes(patcher, active, return_all=True)
        fluxes, precisions = result[:2]
        scratchpad.precision_matrices = precisions
        # update flux values and bounds based on precision matrix.
        for i, b in enumerate(patcher.bandlist):
            f = np.atleast_1d(fluxes[i])
            lo, hi = flux_bounds(f, config.flux_prior_expansion,
                                 precisions=precisions[i])
            bounds[b][:, 0] = lo
            bounds[b][:, 1] = hi
            active[b] = f
        patcher.set_scene(active)
        scratchpad.active_postlinear = active.copy()
        scratchpad.bounds_postlinear = bounds.copy()
        scratchpad.q_postlinear = patcher.scene.get_all_source_params().copy()

    return (out, step, stats), (region, active, fixed, bounds), model


def sample(patcher, scene, config, logger,
           cov=None, scratchpad=argparse.Namespace()):

    region, active, fixed, bounds = scene

    # We rebuild the patch and model to zero-out any side-effects of optimization
    patcher.build_patch(region, None, allbands=config.bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed, bounds=bounds)
    scratchpad.q_sampling = q.copy()
    if cov is None:
        cov = np.eye(len(q))
    out, step, stats = run_lmc(model, q.copy(),
                               n_draws=config.sampling_draws,
                               warmup=config.warmup,
                               z_cov=cov, full=True,
                               weight=max(10, active["n_iter"].min()),
                               discard_tuned_samples=False,
                               max_treedepth=config.max_treedepth,
                               progressbar=config.progressbar)

    return (out, step, stats), (region, active, fixed, bounds), model
