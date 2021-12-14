# -*- coding: utf-8 -*-

import os, json, logging
import numpy as np
import h5py

from numpy.linalg import LinAlgError

from forcepho.fitting import run_opt, run_lmc, optimize_fluxes
from forcepho.utils import write_residuals
from forcepho.superscene import flux_bounds

__all__ = ["write_to_disk",
           "accomplish_task", "optimization", "sampling",
           "lsq_optimize"]


def write_to_disk(out, outroot, model, config, residual=None):

    # --- write the chain and meta-data for this task ---
    outfile = f"{outroot}_samples.h5"
    try:
        out.config = json.dumps(vars(config))
    except(TypeError):
        pass
    out.dump_to_h5(outfile)

    # --- Write image data and residuals if requested ---
    if config.write_residuals:
        outfile = f"{outroot}_residuals.h5"
        if residual is None:
            q = out.chain[-1, :]  # last position in chain
            residual = model.residuals(q)
        write_residuals(model.patch, outfile, residuals=residual)


def accomplish_task(patcher, task, config=None, logger=None,
                    method=None):

    # --- unpack all the task variables ---
    region = task['region']
    active, fixed, big = task['active'], task['fixed'], task["big"]
    bounds, cov = task['bounds'], task['cov']
    bands, shape_cols = task['bands'], task['shape_cols']
    sourceID, taskID = active[0]["source_index"], task["taskID"]
    del task

    # --- log receipt ---
    logger.info(f"Received RA {region.ra}, DEC {region.dec} with tag {taskID}")
    logger.info(f"Tag {taskID} has {len(active)} sources seeded on source index {sourceID}")
    if fixed is None:
        logger.info("No fixed sources")
    if big is None:
        logger.info("No big sources")

    # --- get pixel data and metadata ---
    patcher._dirty_data = False
    patcher.build_patch(region, None, allbands=bands, tweak_background=config.tweak_background)
    if patcher.background_offsets is not None:
        logger.info(f"background offsets applied: {patcher.background_offsets}")
    else:
        logger.info(f"No background offsets applied.")
    logger.info(f"Prepared patch with {patcher.npix} pixels.")
    if config.sampling_draws == 0:
        return patcher

    # --- Prepare model ---
    model, q = patcher.prepare_model(active=active, fixed=fixed, big=big,
                                     bounds=bounds, shapes=shape_cols,
                                     big_scene_kwargs=getattr(config, "big_scene_kwargs", {}))
    model.ndof = patcher.npix
    if config.add_barriers:
        logger.info("Adding edge prior to model.")
        from forcepho.priors import ExpBeta
        model._lnpriorfn = ExpBeta(model.transform.lower, model.transform.upper)

    # --- Do the work ---
    weight = max(10, active["n_iter"].min())
    out, step, stats = method(model, q.copy(), cov=cov, weight=weight, disp=False,
                              config=config, logger=logger)

    # --- pack up ---
    final, covs = out.fill(region, active, fixed, model, bounds=bounds, big=big,
                           step=step, stats=stats, patchID=taskID)

    # --- add extra linear optimization? ---
    if getattr(config, "linear_optimize", False):
        logger.info(f"Doing linear optimization of fluxes")
        final, bounds, out = lsq_optimize(patcher, start=final, out=out,
                                          logger=logger, config=config, taskID=taskID)
        out.final = final

    # --- write ---
    outroot = os.path.join(config.patch_dir, f"patch{taskID}")
    logger.info(f"Writing to {outroot}*")
    write_to_disk(out, outroot, model, config)

    # --- develop the payload ---
    payload = dict(out=out, final=final, covs=covs, bounds=bounds)

    return payload


def optimization(model, start, cov=None, config=None, logger=None, disp=False, **extras):

    logger.info(f"Beginning optimization.")

    # --- Run fit/optimization ---
    model.sampling = False
    opt, scires = run_opt(model, start.copy(), jac=bool(config.use_gradients),
                          disp=disp, gtol=config.gtol)

    # --- clean up ---
    model.sampling = True
    model._lnpriorfn = None
    logger.info(f"Optimization complete ({model.ncall} calls).")
    logger.info(f"{scires.message}")

    # --- develop output ---
    out, step, stats = opt, None, None
    out.linear_optimized = False

    return out, step, stats


def sampling(model, start, cov=None, config=None, logger=None, weight=10, **extras):

    logger.info(f"Beginning sampling with weight={weight}.")

    try:
        discard_tuning = bool(getattr(config, "discard_tuning", True))
        out, step, stats = run_lmc(model, start.copy(),
                                   n_draws=config.sampling_draws,
                                   warmup=config.warmup,
                                   z_cov=cov, full=config.full_cov,
                                   weight=weight,
                                   discard_tuned_samples=discard_tuning,
                                   max_treedepth=config.max_treedepth,
                                   progressbar=getattr(config, "progressbar", False))
    except ValueError as e:
        logger.error(f"Error at constrained parameter q={start}")
        logger.error(f"Error at unconstrained parameter"
                     f"z={model.transform.inverse_transform(start)}")
        raise e

    logger.info(f"Sampling complete ({model.ncall} calls), preparing output.")

    return out, step, stats


def lsq_optimize(patcher, start=None, out=None, factor=3,
                 logger=None, config=None, taskID=0):
    """Conduct linear optimization of fluxes (conditional on shape)
    """

    design = os.path.join(config.patch_dir, f"patch{taskID}_design.h5")

    preop = start.copy()
    preop_bounds = out.bounds.copy()
    result = optimize_fluxes(patcher, preop, return_all=True)
    if type(result[0]) is int:
        logger.error(f"Error during linear optimization of band {patcher.bandlist[result[0]]}, skipping!!")
        abort_lsq(design, result, patcher, logger=logger, badband=result[0])
        return start, preop_bounds, out

    # update fluxes and bounds
    fluxes, precisions = result[:2]
    bounds = preop_bounds.copy()
    final = preop.copy()
    # FIXME: clean this up!
    for i, b in enumerate(patcher.bandlist):
        f = np.atleast_1d(fluxes[i])
        if np.any(~np.isfinite(f)):
            logger.info(f"Invalid optimized flux in band {b} with fluxes {f}, skipping!")
            abort_lsq(design, result, patcher, logger=logger, badband=i)
            return start, preop_bounds, out

        lo, hi = flux_bounds(f, factor, precisions=precisions[i])
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(lo)):
            logger.info(f"could not create valid bounds for band {b} with fluxes {f}")
            continue

        if np.any(lo >= f):
            logger.info(f"low bound > flux for band {b} with\nfluxes: {f}\nlower: {lo}")
            continue

        if np.any(hi <= f):
            logger.info(f"upper bound <= flux for band {b} with\nfluxes: {f}\nupper: {hi}")
            continue

        # update this band
        final[b] = f
        bounds[b][:, 0] = lo
        bounds[b][:, 1] = hi

    # --- cache the starting values and bounds ---
    out.linear_optimized = True
    out.preop_bounds = preop_bounds
    out.preop = preop
    out.bounds = bounds
    out.postop = final
    out.precisions = np.array(precisions)

    # Now reset the data and such on the GPU
    patcher._dirty_data = False
    _, scene = patcher.subtract_fixed(fixed=out.fixed, active=final, big=out.big,
                                      big_scene_kwargs=getattr(config, "big_scene_kwargs", {}))
    # add final position to chain
    q = scene.get_all_source_params()
    out.chain = np.concatenate([out.chain, np.atleast_2d(q)])

    return final, bounds, out


def abort_lsq(fn, matrices, patcher, logger=None, badband=None):

    logger.info(f"Writing design matrix to {fn}")

    i, e, Xes, ys, ws = matrices
    xpix = patcher.split_band("xpix")
    ypix = patcher.split_band("ypix")

    pixel_lists = dict(X=Xes, w=ws, y=ys,
                       xpix=xpix, ypix=ypix)

    with h5py.File(fn, "a") as h5:
        h5.attrs["failed band"] = patcher.bandlist[badband]
        for i, b in enumerate(patcher.bandlist):
            g = h5.create_group(b)
            for k, v in pixel_lists.items():
                if v is not None:
                    g.create_dataset(k, data=v[i])
