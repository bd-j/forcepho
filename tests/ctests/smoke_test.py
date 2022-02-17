#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate a forcepho galaxy model using fast code and compare to the given
reference image.
"""


import os, sys, glob, shutil, time
import argparse, logging, json
import numpy as np

from astropy.io import fits

import forcepho
from forcepho.patches import FITSPatch, CPUPatchMixin, SimplePatch
from forcepho.utils import NumpyEncoder, read_config, write_residuals
from forcepho.sources import Galaxy

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


class SimpleCPUPatch(FITSPatch, CPUPatchMixin):
    pass


if HASGPU:
    Patch = SimplePatch
    kernel = "gpukernel"
else:
    Patch = SimpleCPUPatch
    kernel = "cppkernel"


logging.basicConfig(level=logging.DEBUG)


def test_ingredients(config_file="./verification_config.yml",
                     reference_image="./reference-2021Nov30_f200w_sersic=2.2_rhalf=0.10.fits",
                     bandlist=["F200W"],
                     ):

    logger = logging.getLogger('ingredients')

    args = argparse.Namespace()
    args.config_file = config_file
    args.reference_image = reference_image
    args.bandlist = bandlist

    # --- Configure ---
    config = read_config(args.config_file, args)
    config.patch_dir = os.path.join(config.outbase, "patches")
    [os.makedirs(a, exist_ok=True) for a in (config.outbase, config.patch_dir)]
    _ = shutil.copy(config.config_file, config.outbase)

    # --- get the reference data, image and catalog ---
    config.fitsfiles = [config.reference_image]
    n = config.fitsfiles[0]
    hdul = fits.open(n)
    active, hdr = np.array(hdul[-1].data), hdul[-1].header
    bands = hdr["FILTERS"].split(",")
    bands = [b.upper() for b in bands]
    logger.info("Configured")

    # --- Build the patch with pixel data ---
    patcher = Patch(psfstore=config.psfstorefile,
                    splinedata=config.splinedatafile,
                    fitsfiles=config.fitsfiles,
                    return_residual=True)
    logger.info("Instantiated patcher.")

    patcher.build_patch(region=None, allbands=bands)
    logger.info("Built patch.")

    # --- Build the model with scene ---
    shape_cols = Galaxy.SHAPE_COLS
    model, q = patcher.prepare_model(active=active, shapes=shape_cols)
    logger.info("Prepped model.")

    return config, hdul, model, q


def test_verify(write=True):
    """Compare model image to a reference image by comparing residuals.
    """

    # --- Logger
    logger = logging.getLogger('verify')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()

    # --- evaluate the model ---
    residuals = model.residuals(q)
    model_image = patcher.data - residuals
    logger.info("Computed residuals.")

    # check valid pixels
    valid = patcher.xpix >= 0
    if not np.allclose(residuals[0][valid], 0, atol=5e-7):
        diff = np.nanmax(np.abs(residuals[0][valid]))
        raise ValueError(f"Model did not match reference image: max abs diff = {diff}")
    else:
        logger.info("Model matches reference, yay!")

    # --- write the results ---
    if write:
        vers = forcepho.__version__
        ts = time.strftime("%Y%b%d", time.localtime())
        out = f"output/verification_residuals_{kernel}_v{vers}.h5"
        write_residuals(patcher, out, residuals=[residuals])
        logger.info(f"Wrote residuals to {out}")

    hdul.close()


def test_chi2_gradients():
    # --- Logger
    logger = logging.getLogger('verify')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()
    patcher = model.patch
    valid = patcher.xpix >= 0

    # --- evaluate model ---
    z = model.transform.inverse_transform(q)
    lnp, lnp_grad = model.lnprob_and_grad(z)
    logger.info("Computed likelihood.")

    # check that chi2 close to zeros
    chi2 = ((patcher.residual * patcher.ierr)**2).sum()
    assert np.allclose(0, chi2[valid].sum(), atol=2e-7)

    # this actually how chi2 is calculated by the kernels
    chi2 -= ((patcher.data*patcher.ierr)**2).sum()
    assert np.allclose(chi2[valid].sum(), -2 * lnp, rtol=1e-7, atol=1e-7)

    # here's the dchi2_dim image
    dchi2_ddata = 2 * patcher.residual * patcher.ierr**2

    for hdu in hdul[1:-1]:
        param = hdu.header["BUNIT"]
        ddata_dp = hdu.data.T.reshape(-1)
        dchi2_dp = dchi2_ddata[valid] * ddata_dp
        grad = np.sum(dchi2_dp)



def test_gradients():

    #residuals = patcher.residual
    im = np.zeros([63, 32])
    valid = patcher.xpix >= 0
    im[patcher.xpix[valid].astype(int), patcher.ypix[valid].astype(int)] = residuals[0][valid]

    pass

if __name__ == "__main__":
    # --- Logger
    logger = logging.getLogger('verify')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()
    patcher = model.patch
    valid = patcher.xpix >= 0

    # --- evaluate model ---
    z = model.transform.inverse_transform(q)
    lnp, lnp_grad = model.lnprob_and_grad(z)
    logger.info("Computed likelihood.")
