# -*- coding: utf-8 -*-

import sys
from itertools import product
import os, argparse
import numpy as np
import h5py

import galsim
from astropy.io import fits

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to


if __name__ == "__main__":

    parser = get_parser()
    parser.set_defaults(nx=128, ny=128,
                        sigma_psf=[1.5],
                        snr=15,
                        rhalf=[0.2, 0.2])
    parser.add_argument("--n_dither", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--basename", type=str, default="dither")
    config = parser.parse_args()
    print(config)
    assert len(config.scales) == 1
    assert len(config.bands) == 1

    try:
        os.remove(config.psfstore)
    except:
        pass

    # Set up the dither pattern
    # This is from NIRCAM
    dithers = [(0, 0),
               (0.0352, 0.1387),
               (0.0640, 0.3093),
               (0.1067, 0.0640),
               (0.1707, 0.1707),
               (0.2027, 0.3413),
               (0.2777, 0.0320),
               (0.3093, 0.1387),
               (0.3414, 0.3093)]
    dithers = np.array(dithers) / config.scales[0]
    if config.n_dither:
        dithers = dithers[:config.n_dither]

    # Make the images
    stamps = [make_stamp(config.bands[0], nx=config.nx, ny=config.ny,
                         scale=config.scales[0], dither=(dx, dy))
              for dx, dy in dithers]

    # --- Set the scene in the image(s) ---
    scene = make_scene(stamps, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic)

    # --- Make and store the psf ---
    psf = get_galsim_psf(config.scales[0], sigma_psf=config.sigma_psf[0])
    make_psfstore(config.psfstore, config.bands[0], config.sigma_psf[0], nradii=9)

    # --- Render the scene in each stamp ---
    images = []
    for i, stamp in enumerate(stamps):
        images.append(galsim_model(scene, stamp, psf=psf))

    # --- compute uncertainty ---
    noise_per_pix = compute_noise_level(scene, config)

    # --- write the dither images ---
    os.makedirs(config.outdir, exist_ok=True)
    for i, stamp in enumerate(stamps):
        band = stamp.filtername
        hdul, wcs = stamp.to_fits()
        hdul[0].header["EXTNAME"] = "SCI"
        hdul[1].header["EXTNAME"] = "ERR"
        hdr = hdul[0].header
        hdr["FILTER"] = band.upper()
        hdr["SNR"] = config.snr
        hdr["DFRAC"] = config.dist_frac
        hdr["NOISED"] = config.add_noise
        hdr["PSFSIG"] = config.sigma_psf[0]

        im = images[i]
        unc = np.ones_like(im) * noise_per_pix
        noise = np.random.normal(0, noise_per_pix, size=im.shape)
        if config.add_noise:
            im += noise

        out = os.path.join(f"{config.outdir}", f"{config.basename}_{i:02.0f}.fits")
        write_fits_to(out, im, unc, hdr, config.bands, noise=noise, scene=scene)
