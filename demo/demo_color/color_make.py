#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, argparse
import numpy as np

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to


if __name__ == "__main__":

    # --- Configure ---
    parser = get_parser()
    parser.set_defaults(bands=["BLUE", "RED"],
                        scales=[0.03, 0.06],
                        sigma_psf=[1.5, 2.25],
                        rhalf=[0.2, 0.2],
                        psfstore="./color_gausspsf.h5")
    parser.add_argument("--outdir", type=str, default=".")
    config = parser.parse_args()

    # --- Are we doing one or two sources ---
    ext = ["single", "pair"]
    nsource = len(config.rhalf)

    # --- Where does the PSF info go? ---
    try:
        os.remove(config.psfstore)
    except:
        pass

    # --- Make the images ---
    stamps = [make_stamp(band.upper(), nx=config.nx, ny=config.ny, scale=scale)
              for band, scale in zip(config.bands, config.scales)]

    # --- Set the scene in the image ---
    scene = make_scene(stamps, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic)

    # render the scene in each stamp
    # also store the psf
    images = []
    for i in range(2):
        band, scale, sigma = config.bands[i].upper(), config.scales[i], config.sigma_psf[i]
        psf = get_galsim_psf(scale, psf_type="simple", sigma_psf=sigma)
        images.append(galsim_model(scene, stamps[i], psf=psf))
        make_psfstore(config.psfstore, band, sigma, nradii=9)

    # --- compute uncertainty ---
    noise_per_pix = compute_noise_level(scene, config)

    # --- write the test images ---
    for i, stamp in enumerate(stamps):
        band = stamp.filtername
        hdul, wcs = stamp.to_fits()
        hdr = hdul[0].header
        hdr["FILTER"] = band.upper()
        hdr["SNR"] = config.snr
        hdr["DFRAC"] = config.dist_frac
        hdr["NOISED"] = config.add_noise

        im = images[i]
        unc = np.ones_like(im) * noise_per_pix[i]
        noise = np.random.normal(0, noise_per_pix[i], size=im.shape)
        if config.add_noise:
            im += noise

        os.makedirs(config.outdir, exist_ok=True)
        out = f"{config.outdir}/{band.lower()}_{ext[nsource -1]}.fits"
        write_fits_to(out, im, unc, hdr, config.bands,
                      noise=noise, scene=scene)
