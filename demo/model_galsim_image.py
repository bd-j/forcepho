#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""model_test_image.py - generate image with forcepho for a particular input scene
"""

import sys, os
import numpy as np

import argparse
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.proposal import Proposer
from forcepho.patches import JadesPatch
from forcepho.region import RectangularRegion

from forcepho.utils import Logger, rectify_catalog, read_config

try:
    import pycuda
    import pycuda.autoinit
except:
    print("NO PYCUDA")


def parse_image(hdr, cat, n_gal=16):
    """The image is small, so we will just predict the entire image for each set of sources.
    We are limited in the number of sources by GPU memory, currently the max is ~20
    """
    wcs = WCS(hdr)
    x = [0, hdr["NAXIS1"], hdr["NAXIS1"], 0]
    y = [0, 0, hdr["NAXIS2"], hdr["NAXIS2"]]
    ra, dec = wcs.all_pix2world(x, y, 0)
    region = RectangularRegion(ra.min(), ra.max(), dec.min(), dec.max())

    n_patch = int(np.ceil(len(cat) / n_gal))
    actives = np.array_split(cat, n_patch)
    return region, actives


def get_model_cpu(region, active, patcher):
    patcher.build_patch(region, active, allbands=config.bandlist)
    patcher.scene.set_all_source_params(q)
    stamps = patch_to_stamps(patcher)

    residuals = []
    for stamp in stamps:
        im, grad = make_image(patcher.scene, stamp)
        residuals.append(stamp.pixel_values - im)

    result = {"residual": residuals,
              "reference_coordinates": patcher.patch_reference_coordinates.copy(),
              "parameter_vector": q,
              "proposal": prop}
    return result


def get_model_gpu(active, patcher):
    # --- Build the things
    patcher.pack_meta(active)
    gpu_patch = patcher.send_to_gpu()
    proposer = Proposer(patcher)

    # --- Get parameter vector and proposal
    q = patcher.scene.get_all_source_params()
    prop = patcher.scene.get_proposal()

    # --- evaluate proposal, including residuals
    proposer.patch.return_residual = True
    out = proposer.evaluate_proposal(prop)

    result = {"residual": out[-1],
              "reference_coordinates": patcher.patch_reference_coordinates.copy(),
              "parameter_vector": q,
              "proposal": prop
              }

    return result


if __name__ == "__main__":

    # read command lines
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="galsim.yml")
    parser.add_argument("--outbase", type=str, default="output")
    args = parser.parse_args()

    # read config file
    config = read_config(args.config_file, args)

    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)

    iexp = 0

    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    hdrs = [hdr for band in patcher.metastore.headers.keys()
            for hdr in list(patcher.metastore.headers[band].values())[:1]]
    region, actives = parse_image(hdrs[0], cat, n_gal=config.maxactive_per_patch)
    patcher.build_patch(region, None, allbands=config.bandlist)

    data = patcher.split_pix("data")[iexp]
    xpix = patcher.split_pix("xpix")[iexp].astype(int)
    ypix = patcher.split_pix("ypix")[iexp].astype(int)
    band, exp = patcher.epaths[iexp].split("/")
    hdr = patcher.metastore.headers[band][exp]

    im = np.zeros([hdr["NAXIS1"], hdr["NAXIS2"], len(actives)])

    #sys.exit()

    for i, active in enumerate(actives):
        # TODO: This could work by successive data/residual swaps on the GPU
        result = get_model_gpu(active, patcher)
        model = data - result["residual"][0]
        im[xpix, ypix, i] += model

    # write out the sum
    outfile = os.path.join(args.outbase, exp.replace("noisy", "fmodel") + ".fits")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    image = im.sum(axis=-1).T
    with fits.HDUList(fits.PrimaryHDU(image)) as hdul:
        hdul[0].header.update(hdr[6:])
        hdul[0].header["NOISE"] = 0
        hdul.writeto(outfile, overwrite=True)
