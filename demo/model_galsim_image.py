#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""model_test_image.py - generate image with forcepho for a particular input scene
"""

import sys, os
import numpy as np

import argparse
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.reconstruction import Reconstructor
from forcepho.patches import JadesPatch
from forcepho.region import RectangularRegion

from forcepho.utils import Logger, rectify_catalog, read_config

try:
    import pycuda
    import pycuda.autoinit
except:
    print("NO PYCUDA")


def parse_image(hdr):
    """The image is small, so we will just predict the entire image for each set of sources.
    We are limited in the number of sources by GPU memory, currently the max is ~20
    """
    wcs = WCS(hdr)
    x = [0, hdr["NAXIS1"], hdr["NAXIS1"], 0]
    y = [0, 0, hdr["NAXIS2"], hdr["NAXIS2"]]
    ra, dec = wcs.all_pix2world(x, y, 0)
    region = RectangularRegion(ra.min(), ra.max(), dec.min(), dec.max())

    return region


if __name__ == "__main__":

    # read command lines
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="galsim_config.yml")
    parser.add_argument("--outbase", type=str, default="output")
    args = parser.parse_args()

    # read config file
    config = read_config(args.config_file, args)

    # Make patcher and reconstroctor
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)
    recon = Reconstructor(patcher, MAXSOURCES=config.maxactive_per_patch)

    # read the parameter catalog, build a region
    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    hdrs = [hdr for band in patcher.metastore.headers.keys()
            for hdr in list(patcher.metastore.headers[band].values())[:1]]
    region = parse_image(hdrs[0])

    # Reconstruct the model image
    recon.fetch_data(region, bands)
    model = recon.model_data(cat)
    iexp = 0
    image, hdr, epath = recon.get_model_image(model, iexp=iexp)

    # write out the model image
    outfile = os.path.basename(epath).replace("noisy", "forcetruth") + ".fits"
    outfile = os.path.join(args.outbase, outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with fits.HDUList(fits.PrimaryHDU(image.T)) as hdul:
        hdul[0].header.update(hdr[6:])
        hdul[0].header["NOISE"] = 0
        hdul.writeto(outfile, overwrite=True)
