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
    parser.set_defaults(bands=["CLEAR"],
                        scales=[0.03],
                        sigma_psf=[2.5],
                        rhalf=[0.2])
    parser.add_argument("--outdir", type=str, default="./")
    parser.add_argument("--snrlist", type=float, nargs="*", default=[10, 30, 100])
    config = parser.parse_args()

    # --- Where does the PSF info go? ---
    try:
        os.remove(config.psfstore)
    except:
        pass

    band, sigma, scale = config.bands[0], config.sigma_psf[0], config.scales[0]

    # Make the image
    stamp = make_stamp(band, scale=scale, nx=config.nx, ny=config.ny)

    # Set the scene in the image
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic, pa=0.0)

    # render the scene
    psf = get_galsim_psf(scale, sigma_psf=sigma)
    make_psfstore(config.psfstore, band, sigma, nradii=9)
    im = galsim_model(scene, stamp, psf=psf)

    for snr in config.snrlist:
        config.snr = snr
        noise_per_pix = compute_noise_level(scene, config)

        unc = np.ones_like(im)*noise_per_pix
        noise = np.random.normal(0, noise_per_pix, size=im.shape)
        if config.add_noise:
            im += noise

        # --- write the test image ---
        hdul, wcs = stamp.to_fits()
        hdr = hdul[0].header
        hdr["FILTER"] = band
        hdr["SNR"] = config.snr
        hdr["DFRAC"] = config.dist_frac
        hdr["NOISED"] = config.add_noise

        out = os.path.join(f"{config.outdir}", f"single_snr{snr:03.0f}.fits")
        write_fits_to(out, im, unc, hdr, config.bands, noise=noise, scene=scene)


