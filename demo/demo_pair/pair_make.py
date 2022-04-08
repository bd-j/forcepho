# -*- coding: utf-8 -*-

import sys, os, argparse
import numpy as np

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--outname", type=str, default="./pair.fits")
    config = parser.parse_args()
    config.bands = [b.upper() for b in config.bands]
    assert len(config.bands) == 1
    try:
        os.remove(config.psfstore)
    except:
        pass

    # --- Make the image ---
    stamp = make_stamp(config.bands[0], scale=config.scales[0],
                       nx=config.nx, ny=config.ny)

    # --- Set the scene in the image ---
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic)

    # --- render the scene ---
    band, scale, sigma = config.bands[0], config.scales[0], config.sigma_psf[0]
    psf = get_galsim_psf(scale, sigma_psf=sigma)
    im = galsim_model(scene, stamp, psf=psf)
    make_psfstore(config.psfstore, band, sigma, nradii=9)

    # --- compute uncertainty ---
    noise_per_pix = compute_noise_level(scene, config)

    # --- write the test image ---
    hdul, wcs = stamp.to_fits()
    hdr = hdul[0].header
    hdr["FILTER"] = band
    hdr["SNR"] = config.snr
    hdr["DFRAC"] = config.dist_frac
    hdr["NOISED"] = config.add_noise
    hdr["PSFSIG"] = sigma

    unc = np.ones_like(im) * noise_per_pix
    noise = np.random.normal(0, noise_per_pix, size=im.shape)
    if config.add_noise:
        im += noise

    os.makedirs(os.path.dirname(config.outname), exist_ok=True)
    write_fits_to(config.outname, im, unc, hdr, config, noise=noise, scene=scene)