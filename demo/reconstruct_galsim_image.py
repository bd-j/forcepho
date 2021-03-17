#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""reconstruct_test_image.py - Generate an image with forcepho using a set of
chains from different patches to define the scene.
"""

from argparse import ArgumentParser
import sys, os, glob
import numpy as np

import argparse
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.reconstruction import Reconstructor
from forcepho.patches import JadesPatch
from forcepho.region import RectangularRegion

from forcepho.fitting import Result
from forcepho.utils import rectify_catalog, read_config

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU=False


def reconstruct_scene(results_files_list):

    # --- pull the best fit model from each chain ---
    bests, lnps = [], np.zeros(len(results_files_list))
    for i, fn in enumerate(results_files_list):
        result = Result(fn)
        ind = np.argmax(result.stats["model_logp"])
        lnps[i] = result.stats["model_logp"][ind]
        best.append(result.get_sample_cat(ind))

    bests = np.concatenate(bests)

    # -- choose a particular model for duplicates ---
    order = np.argsort(lnps)[::-1]
    bests = bests[order]
    _, inds = np.unique(bests["source_index"], return_index=True)
    return bests, inds


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

    # Configure
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default="galsim.yml")
    parser.add_argument("--run_id", type=str, default="run1")
    args = parser.parse_args()
    config = read_config(args.config_file, args)
    config.outbase = config.outbase.replace("run1", config.run_id)
    config.patch_dir = config.patch_dir.replace("run1", config.run_id)
    bands = config.bandlist

    # Build patcher and reconstructor
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)
    recon = Reconstructor(patcher, MAXSOURCES=config.maxactive_per_patch)

    # Find patch results and reconstruct scene
    files = glob.glob(os.path.join(args.patch_dir, "patch*_samples.h5"))
    fullcat, inds = reconstruct_scene(files)
    cat = fullcat[inds]

    # Get a region that encompasses all the images
    hdrs = [hdr for band in patcher.metastore.headers.keys()
            for hdr in list(patcher.metastore.headers[band].values())[:1]]
    region = parse_image(hdrs[0])

    recon.fetch_data(region, bands)
    model = recon.model_data(fullcat)
    iexp = 0
    image, hdr, epath = recon.get_model_image(model, iexp=iexp)

    if not HASGPU:
        sys.exit()

    # write out the sum
    outfile = os.path.basename(epath).replace("noisy", "forcebest") + ".fits"
    outfile = os.path.join(args.output_dir, outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with fits.HDUList(fits.PrimaryHDU(image.T)) as hdul:
        hdul[0].header.update(hdr[6:])
        hdul[0].header["NOISE"] = 0
        hdul.writeto(outfile, overwrite=True)
