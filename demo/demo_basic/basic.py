#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin
from forcepho.superscene import LinkedSuperScene
from forcepho.utils import NumpyEncoder, write_to_disk
from forcepho.fitting import run_lmc

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to


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
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--write_residuals", type=int, default=1)
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=2048)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=1)
    config = parser.parse_args()

    config.tag = make_tag(config)
    config.outdir = os.path.join("./tests/", config.tag)
    os.makedirs(config.outdir, exist_ok=True)
    outroot = os.path.join(config.outdir, config.tag)
    config.psfstore = f"{outroot}_psf.h5"
    config.image_name = f"{outroot}_data.fits"
    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]

    # ---------------------
    # --- Make the data ---
    # make empty stamp and put scene in it
    stamp = make_stamp(band, scale=scale, nx=config.nx, ny=config.ny)
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic, pa=0.0)
    # Render the scene in galsim, including PSF
    psf = get_galsim_psf(scale, sigma_psf=sigma)
    if os.path.exists(config.psfstore):
        os.remove(config.psfstore)
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

    # --------------------
    # --- Fit the data ---
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

    # prepare model and data, and sample
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

    # Check results back in and end
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=0)
    write_to_disk(out, outroot, model, config)
    sceneDB.checkin_region(final, fixed, config.sampling_draws,
                           block_covs=covs, taskID=0)
    sceneDB.writeout()
