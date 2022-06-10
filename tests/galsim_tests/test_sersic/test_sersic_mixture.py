#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

import matplotlib.pyplot as pl
from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
from forcepho.superscene import LinkedSuperScene
from forcepho.utils import write_to_disk, NumpyEncoder
from forcepho.fitting import run_lmc
from forcepho.postprocess import Samples

from test_utils import get_parser, get_grid_params
from test_utils import make_stamp, make_scene
from test_utils import get_galsim_psf, galsim_model, compute_noise_level
from test_utils import make_psfstore, write_fits_to

from test_plot import plot_trace, plot_corner, plot_residual
from test_plot import make_catalog, compare_parameters, compare_apflux

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


__all__ = ["make_image",
           "fit_image"]


if HASGPU:
    class Patcher(FITSPatch, GPUPatchMixin):
        pass
else:
    class Patcher(FITSPatch, CPUPatchMixin):
        pass


def make_tag(config):  #, params, names):
    # this could be programmatic, but it would be annoying to write.
    tag = f"sersic{config.sersic[0]:.1f}_rhalf{config.rhalf[0]:.3f}_q{config.q[0]:01.2f}"
    tag += f"_fwhm{config.fwhm:01.1f}_snr{config.snr:03.0f}_noise{config.add_noise:.0f}"
    return tag


def make_all_tags(grid, config):
    from argparse import Namespace
    pars = Namespace(add_noise=config.add_noise,
                     fwhm=config.fwhm)
    tags = []
    for row in grid:
        [setattr(pars, p, [row[p]]) for p in ["sersic", "rhalf", "q"]]
        [setattr(pars, p, row[p]) for p in ["snr"]]
        tags.append(make_tag(pars))
    return tags


def make_image(config):
    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]
    # make empty stamp and put scene in it
    stamp = make_stamp(band, scale=scale, nx=config.nx, ny=config.ny)
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       q=config.q, pa=config.pa,
                       rhalf=config.rhalf, sersic=config.sersic)
    # Render the scene in galsim, including PSF
    psf = get_galsim_psf(scale, sigma_psf=sigma)
    try:
        make_psfstore(config.psfstore, band, sigma, nradii=9)
    except(ValueError):
        pass
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
    hdul.close()


def fit_image(config):
    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]

    # build the scene server
    cat = fits.getdata(config.image_name, -1)
    bands = fits.getheader(config.image_name, -1)["FILTERS"].split(",")
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "final_scene.fits"),
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=dict(n_pix=1.0, rhalf_range=(0.03, 1.0), sersic_range=(0.8, 5.0)),
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

    # Check results back in and end and write everything to disk
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=0)
    write_to_disk(out, config.outroot, model, config)
    sceneDB.checkin_region(final, fixed, config.sampling_draws,
                           block_covs=covs, taskID=0)
    sceneDB.writeout()


if __name__ == "__main__":

    print(f"HASGPU={HASGPU}")

    # ------------------
    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        sigma_psf=[2.5],
                        rhalf=[0.2],
                        sersic=[2.0])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dir", type=str, default="./output")
    parser.add_argument("--test_grid", type=str, default="./test_sersic_grid.yml")
    parser.add_argument("--start", type=int, default=0)
    # I/O
    parser.add_argument("--splinedatafile", type=str, default="./sersic_splinedata_large.h5")
    parser.add_argument("--write_residuals", type=int, default=1)
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=2048)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=0)
    config = parser.parse_args()

    # copy the config data
    os.makedirs(config.dir, exist_ok=True)
    with open(f"{config.dir}/config.json", "w") as cfg:
        json.dump(vars(config), cfg, cls=NumpyEncoder)

    # --- Set up the grid ---
    params = get_grid_params(config, start=config.start)
    # write the input grid
    gname = config.test_grid.replace(".yml", ".fits")
    outgridname = os.path.join(config.dir, os.path.basename(gname))
    fits.writeto(outgridname, params, overwrite=True)
    tags = []

    # loop over grid, generating images and fitting
    for i, param in enumerate(params):
        # set parameters in config
        config.rhalf = [param["rhalf"]]
        config.sersic = [param["sersic"]]
        config.q = [param["q"]]
        config.sigma_psf = [param["fwhm"]/2.355]
        config.snr = param["snr"]
        config.fwhm = param["fwhm"]
        config.pa = 0

        size_img = int(np.clip(20.0*config.rhalf[0] / config.scales[0], 64, 256))
        config.nx = size_img
        config.ny = size_img

        # make directories and names
        config.tag = make_tag(config)
        config.outdir = os.path.join(config.dir, config.tag)
        os.makedirs(config.outdir, exist_ok=True)
        config.outroot = os.path.join(config.outdir, config.tag)
        config.psfstore = f"{config.outroot}_psf.h5"
        config.image_name = f"{config.outroot}_data.fits"

        # ---------------------
        # --- Make the data ---
        if not os.path.exists(config.image_name):
            make_image(config)

        # --------------------
        # --- Fit the data ---
        patchname = f"{config.outroot}_samples.h5"
        if os.path.exists(patchname):
            continue
        fit_image(config)
        tags.append(config.outroot)

        # --------------------
        # --- make figures ---
        title = config.tag.replace("_", ", ")

        tfig, ax = plot_trace(patchname)
        tfig.suptitle(title)
        tfig.tight_layout()
        tfig.savefig(f"{config.outroot}_trace.png", dpi=200)
        pl.close(tfig)

        cfig, caxes = plot_corner(patchname)
        cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
        cfig.savefig(f"{config.outroot}_corner.png", dpi=200)
        pl.close(cfig)

        rfig, raxes, rcb, val = plot_residual(patchname)
        rfig.savefig(f"{config.outroot}_residual.png", dpi=200)
        pl.close(rfig)

    # Make summary plots
    tcat = fits.getdata(outgridname)
    tags = make_all_tags(tcat, config)
    tags = [os.path.join(config.dir, tag, tag) for tag in tags]
    scat = make_catalog(tags, bands=config.bands)
    fits.writeto(os.path.join(config.dir, "ensemble_chains.fits"), scat)

    comp = [("rhalf", "sersic"), ("sersic", "rhalf"), ("q", "fwhm")]
    for show, by in comp:
        fig, axes = compare_parameters(scat, tcat, show, colorby=by)
        fig.savefig(os.path.join(config.dir, f"{show}_comparison.pdf"))
        pl.close(fig)

    fig, axes = compare_apflux(scat, tcat, bands=config.bands)
    fig.savefig(os.path.join(config.dir, "flux_comparison.png"), dpi=200)
