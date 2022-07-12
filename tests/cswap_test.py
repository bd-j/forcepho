""" test_cswap.py - test that residual/data swaps work with the cpu kernel.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
#import logging
from argparse import ArgumentParser, Namespace
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin
from forcepho.superscene import LinkedSuperScene
from forcepho.postprocess import populate_image


class Patcher(FITSPatch, CPUPatchMixin):
    pass


if __name__ == "__main__":

    config = Namespace()
    config.image_name = "./data/test_image/sersic1.0_rhalf0.150_snr050_noise1_data.fits"
    config.psfstore = "./data/test_image/sersic1.0_rhalf0.150_snr050_noise1_psf.h5"
    config.splinedatafile = "./data/sersic_splinedata_large.h5"

    # build the scene server
    cat = fits.getdata(config.image_name, -1)
    bands = fits.getheader(config.image_name, -1)["FILTERS"].split(",")
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile="./final_scene.fits",
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=dict(n_pix=1.0),
                               target_niter=1)

    # load the image data
    patcher = Patcher(fitsfiles=[config.image_name],
                      psfstore=config.psfstore,
                      splinedata=config.splinedatafile,
                      return_residual=True)

    # check out scene & bounds
    region, active, fixed = sceneDB.checkout_region(seed_index=-1)

    # build the patch
    patcher.build_patch(region, None, allbands=bands)
    scenes = [patcher.set_scene(scene)
              for scene in [active, active]]

    # to communicate with the device
    proposer = patcher.get_proposer()

    # Pack first scene and send it with pixel data
    patcher.return_residual = True
    patcher.pack_meta(scenes[0])
    _ = patcher.send_to_device()

    # evaluate first scene
    proposal = scenes[0].get_proposal()
    out = proposer.evaluate_proposal(proposal, patch=patcher, unpack=False)
    chi, dchi_dp, residual = out
    residual_original = residual
    data_original = patcher.data.copy()

    # pack next scene
    patcher.pack_meta(scenes[1])
    # replace on-device data with on-device residual
    # replace meta of previous block with this block
    # on-device residual becomes 'data'
    # on-device data becomes the residual from the first scene
    patcher.swap_on_device()
    data_postswap = patcher.retrieve_array("data")
    residual_postswap = patcher.retrieve_array("residual")

    # Did the swap work?
    assert(data_postswap is not residual_original)
    assert(residual_postswap is not data_original)
    assert np.allclose(residual_original, data_postswap)
    assert np.allclose(data_original, residual_postswap)

    # now if we evaluate again we should get a negative galaxy
    # = data - model - model
    proposal = scenes[1].get_proposal()
    out = proposer.evaluate_proposal(proposal, patch=patcher, unpack=False)
    chi, dchi_dp, residual_new = out
    model = data_original - residual_original
    assert np.allclose(residual_new, data_original - 2 * model)

    sys.exit()
    im, _, _ = populate_image(patcher.xpix, patcher.ypix, residual_new)
    pl.ion()
    fig, ax = pl.subplots()
    ax.imshow(im.T, origin="lower")
