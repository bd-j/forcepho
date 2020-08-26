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

from forcepho.proposal import Proposer
from forcepho.patches import JadesPatch
from forcepho.region import RectangularRegion

from utils import Logger, rectify_catalog, get_results
from config_test import config

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU=False


def reconstruct_scene(results_files_list):
    cat, _, _ = get_results(results_files_list[0])
    #lnps = [r[-1]["model_logp"] for r in res]
    #inds = [np.argmax(lnp) for lnp in lnps]

    # --- pull the best fit model from each chain ---
    bests, lnps = [], np.zeros(len(results_files_list))
    descr = [desc[:2] for desc in cat.dtype.descr]
    descr += [("lnp", np.float)]
    dtype = np.dtype(descr)
    for i, fn in enumerate(results_files_list):
        cat, _, stats = get_results(fn)
        ind = np.argmax(stats["model_logp"])
        lnp = stats["model_logp"][ind]
        lnps[i] = lnp

        best = np.zeros(len(cat), dtype=dtype)
        best["lnp"] = lnp
        for j, row in enumerate(best):
            for col in dtype.names:
                try:
                    row[col] = cat[col][j][ind]
                except(IndexError):
                    row[col] = cat[col][j]
                except(ValueError):
                    pass

        bests.append(best)
    bests = np.concatenate(bests)

    # -- choose a particular model for duplicates ---
    order = np.argsort(bests["lnp"])[::-1]
    bests = bests[order]
    _, inds = np.unique(bests["source_index"], return_index=True)
    return bests, inds


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

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--patch_dir", type=str, default="./output/run3")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.patch_dir, "patch????_results.h5"))
    fullcat, inds = reconstruct_scene(files)
    cat = fullcat[inds]
    bands = config.bandlist

    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)

    iexp = 0
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

    if HASGPU is False:
        sys.exit()

    for i, active in enumerate(actives):
        # TODO: This could work by successive data/residual swaps on the GPU
        result = get_model_gpu(active, patcher)
        model = data - result["residual"][0]
        im[xpix, ypix, i] += model

    # write out the sum
    outfile = os.path.join(args.output_dir, exp.replace("noisy", "forcebest") + ".fits")
    image = im.sum(axis=-1).T
    with fits.HDUList(fits.PrimaryHDU(image)) as hdul:
        hdul[0].header.update(hdr[6:])
        hdul[0].header["NOISE"] = 0
        hdul.writeto(outfile, overwrite=True)
