#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to conduct frocepho sampling after an initial optimization.
"""

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
from forcepho.superscene import LinkedSuperScene, flux_bounds
from forcepho.utils import NumpyEncoder, write_to_disk, make_chaincat, get_sample_cat
from forcepho.fitting import run_lmc, run_opt, optimize_fluxes

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


if HASGPU:
    class Patcher(FITSPatch, GPUPatchMixin):
        pass
else:
    class Patcher(FITSPatch, CPUPatchMixin):
        pass


def make_tag(config):
    tag = f"sersic{config.sersic[0]:.1f}_rhalf{config.rhalf[0]:.3f}"
    tag += f"_snr{config.snr:03.0f}_noise{config.add_noise:.0f}"
    return tag


if __name__ == "__main__":

    # ------------------
    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        sigma_psf=[2.5],
                        rhalf=[0.2],
                        sersic=[2.0])
    parser.add_argument("--tag", type=str, default="")
    # I/O
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--write_residuals", type=int, default=1)
    # optimization
    parser.add_argument("--linear_optimize", type=int, default=0, help="switch;")
    parser.add_argument("--gtol", type=float, default=0.00001)
    parser.add_argument("--add_barriers", type=int, default=1, help="switch;")
    parser.add_argument("--flux_prior_expansion", type=float, default=5)
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=2048)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=1)
    config = parser.parse_args()

    # --------------------------------
    # --- Filenames and loacations ---
    config.tag = make_tag(config)
    os.makedirs(config.outdir, exist_ok=True)
    outroot = os.path.join(config.outdir, config.tag, config.tag)
    config.psfstore = f"{outroot}_psf.h5"
    config.image_name = f"{outroot}_data.fits"

    try:
        os.remove(config.psfstore)
    except:
        pass

    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]

    # place to store various info
    scratchpad = argparse.Namespace()

    # ---------------------
    # --- Make the data ---
    # make empty stamp and put scene in it
    stamp = make_stamp(band, scale=scale, nx=config.nx, ny=config.ny)
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic, pa=0.0)
    # Render the scene in galsim, including PSF
    psf = get_galsim_psf(scale, sigma_psf=sigma)
    make_psfstore(config.psfstore, band, sigma, nradii=9)
    im = galsim_model(scene, stamp, psf=psf)

    # Noisify
    noise_per_pix = compute_noise_level(scene, config)
    unc = np.ones_like(im)*noise_per_pix
    noise = np.random.normal(0, noise_per_pix, size=im.shape)
    if config.add_noise:
        im += noise

    # write the test image
    hdul, wcs = stamp.to_fits()
    hdr = hdul[0].header
    hdr["FILTER"] = band
    hdr["SNR"] = config.snr
    hdr["NOISED"] = config.add_noise
    write_fits_to(config.image_name, im, unc, hdr, config.bands,
                  noise=noise, scene=scene)

    # ------------------------
    # --- Prep for fitting ---
    # build the scene server
    cat = fits.getdata(config.image_name, -1)
    bands = fits.getheader(config.image_name, -1)["FILTERS"].split(",")
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "final_scene.fits"),
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=dict(n_pix=1.0),
                               target_niter=config.sampling_draws)

    # load the image data
    patcher = Patcher(fitsfiles=[config.image_name],
                      psfstore=config.psfstore,
                      splinedata=config.splinedatafile,
                      return_residual=True)

    # check out scene & bounds
    region, active, fixed = sceneDB.checkout_region(seed_index=-1)
    bounds, cov = sceneDB.bounds_and_covs(active["source_index"])

    # --------------------
    # --- Optimization ---
    # prepare model and data
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=sceneDB.shape_cols)
    scratchpad.q_initial = q.copy()

    # get a new start position randomly drawn from prior
    q = np.random.uniform(model.transform.lower, model.transform.upper)
    scratchpad.q_start = q.copy()
    scratchpad.active_start = active.copy()
    scratchpad.bounds_start = bounds.copy()

    # Add steep penalty near prior edges?
    if config.add_barriers:
        from forcepho.priors import ExpBeta
        model._lnpriorfn = ExpBeta(model.transform.lower, model.transform.upper)

    # Do BFGS optimization
    model.sampling = False
    opt, scires = run_opt(model, q.copy(), jac=True, gtol=config.gtol)
    out, step, stats = opt, None, None
    model.sampling = True
    model._lnpriorfn = None

    # get new starting position and catalog
    q = opt.chain[-1]
    postop_chain = make_chaincat(opt.chain, patcher.bandlist, active,
                                 patcher.patch_reference_coordinates)
    active = get_sample_cat(postop_chain, -1, active)
    #model.scene.set_all_source_params(q_start)
    #active = model.scene.to_catalog()
    scratchpad.q_postop = q.copy()
    scratchpad.active_postop = active.copy()
    scratchpad.bounds_postop = bounds.copy()

    # Do linear flux optimization?
    if config.linear_optimize:
        assert HASGPU
        result = optimize_fluxes(patcher, q, return_all=True)
        fluxes, precisions = result[:2]
        # update flux values and bounds based on precision matrix.
        for i, b in enumerate(patcher.bandlist):
            f = np.atleast_1d(fluxes[i])
            lo, hi = flux_bounds(f, config.flux_prior_expansion,
                                 precisions=precisions[i])
            bounds[b][:, 0] = lo
            bounds[b][:, 1] = hi
            active[b] = f
        scratchpad.active_postlinear = active.copy()
        scratchpad.bounds_postlinear = bounds.copy()
        patcher.scene.set_scene(active)
        scratchpad.q_postlinear = patcher.scene.get_all_source_params().copy()

    # ----------------
    # --- sampling ---
    # Note we use the postoptimization values and bounds
    # We rebuild the patch and model to zero-out any side-effects of optimization
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=sceneDB.shape_cols)
    scratchpad.q_sampling = q.copy()
    out, step, stats = run_lmc(model, q.copy(),
                               n_draws=config.sampling_draws,
                               warmup=config.warmup,
                               z_cov=cov, full=True,
                               weight=max(10, active["n_iter"].min()),
                               discard_tuned_samples=False,
                               max_treedepth=config.max_treedepth,
                               progressbar=config.progressbar)

    # Check results back in and end
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=0)
    for k, v in vars(scratchpad).items():
        setattr(out, k, v)
    write_to_disk(out, outroot, model, config)
    print(f"Wrote output to {outroot}*")
    sceneDB.checkin_region(final, fixed, config.sampling_draws,
                           block_covs=covs, taskID=0)
    sceneDB.writeout()
