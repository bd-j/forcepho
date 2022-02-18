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


logging.basicConfig(level=logging.DEBUG)


def test_ingredients(config_file="./verification_config.yml",
                     reference_image="./reference-2021Nov30_f200w_sersic=2.2_rhalf=0.10.fits",
                     bandlist=["F200W"],
                     kernel_type="gpukernel"):

    logger = logging.getLogger('ingredients')

    args = argparse.Namespace()
    args.config_file = config_file
    args.reference_image = reference_image
    args.bandlist = bandlist

    if (kernel_type == "gpukernel") & HASGPU:
        Patch = SimplePatch
        args.kernel_type = "gpukernel"
    else:
        Patch = SimpleCPUPatch
        args.kernel_type = "cppkernel"

    logger.info(f"Using {args.kernel_type} for the test kernel.")

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


def validate(write=True):
    """Compare model image to a reference image by comparing residuals.
    """

    # --- Logger
    logger = logging.getLogger('verify')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()

    # --- evaluate the model ---
    residuals = model.residuals(q, unpack=False)
    model_image = patcher.data - residuals
    logger.info("Computed residuals.")

    # check valid pixels
    valid = patcher.xpix >= 0
    if not np.allclose(residuals[valid], 0, atol=5e-7):
        diff = np.nanmax(np.abs(residuals[valid]))
        raise ValueError(f"Model did not match reference image: max abs diff = {diff}")
    else:
        logger.info("Model matches reference, yay!")

    hdul.close()


def test_chi2():
    # --- Logger
    logger = logging.getLogger('verify_chi2')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()
    hdul.close()
    patcher = model.patch
    valid = patcher.xpix >= 0

    # --- evaluate model ---
    z = model.transform.inverse_transform(q)
    lnp, lnp_grad = model.lnprob_and_grad(z)
    logger.info("Computed likelihood.")

    # --- check that answers are repeatable ---
    model.evaluate(z)  # forces re-evaluation
    assert np.allclose(lnp, model._lnp)
    assert np.allclose(lnp_grad, model._lnp_grad)

    # check that chi2 close to zeros since this is the correct model
    chi2 = ((patcher.residual * patcher.ierr)**2)
    assert np.allclose(0, chi2[valid].sum(), atol=2e-7)

    # this is actually how chi2 is calculated by the kernels
    # check that it matches the returned lnp
    chi2 -= ((patcher.data*patcher.ierr)**2)
    assert np.allclose(chi2[valid].sum(), -2 * lnp, rtol=1e-7, atol=1e-7)


def test_gradients():

    # --- Logger
    logger = logging.getLogger('verify_gradients')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()
    patcher = model.patch
    valid = patcher.xpix >= 0

    # --- check numerical gradients --
    # we need to be away from MLE so gradients aren't ~0
    qoff = q.copy()
    qoff[3:5] = 0.8, 1.0  # changes q and pa
    zoff = model.transform.inverse_transform(qoff)
    delta = [1e-3, 1e-6, 1e-6, 1e-3, 1e-1, 1e-3, 1e-3]
    dlnp, dlnp_num = model.check_grad(zoff, delta=delta)
    # this is pretty weak, but the numerical gradients are super noisy
    assert np.allclose(dlnp, dlnp_num, rtol=1e-1)

    if False:
        # here's the dchi2_dim image
        residual = model.residuals(q, unpack=False)
        dchi2_ddata = -2 * residual * patcher.ierr**2

        # these checks fail because the reference gradient images were produced
        # at the maximum likelihood value which wreaks havoc in the residual and
        # dchi2 precision
        grad = np.zeros(7)
        for i, hdu in enumerate(hdul[1:-1]):
            param = hdu.header["BUNIT"]
            ddata_dp = hdu.data.T.reshape(-1)
            dchi2_dp = dchi2_ddata[valid] * ddata_dp
            grad[i] = -0.5 * np.sum(dchi2_dp)

    hdul.close()


def display():

    # --- Logger
    logger = logging.getLogger('verify')

    # --- ingredients ---
    config, hdul, model, q = test_ingredients()
    hdul.close()
    patcher = model.patch
    valid = patcher.xpix >= 0

    residuals = model.residuals(q)[0]
    x, y = patcher.xpix[valid].astype(int), patcher.ypix[valid].astype(int)
    im = np.zeros([x.max() + 1, y.max() + 1])
    valid = patcher.xpix >= 0
    im[x, y] = residual[valid]

    return im


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./verification_config.yml")
    parser.add_argument("--reference_image", type=str, default="./reference-2021Nov30_f200w_sersic=2.2_rhalf=0.10.fits")
    parser.add_argument("--bandlist", type=str, nargs="*", default=["F200W"])
    parser.add_argument("--kernel_type", type=str, choices=["cppkernel", "gpukernel"], default="gpukernel")
    args = parser.parse_args()
    params = vars(args)

    logger = logging.getLogger('verify')
    config, hdul, model, q = test_ingredients(**params)
    hdul.close()
    patcher = model.patch

    # --- evaluate the model ---
    residuals = model.residuals(q, unpack=False)
    model_image = patcher.data - residuals
    logger.info("Computed residuals.")

    # --- write the results ---
    if True:
        vers = forcepho.__version__
        ts = time.strftime("%Y%b%d", time.localtime())
        out = f"output/verification_residuals_{config.kernel_type}_v{vers}.h5"
        write_residuals(patcher, out, residuals=[residuals])
        logger.info(f"Wrote residuals to {out}")

    # check valid pixels
    valid = patcher.xpix >= 0
    if not np.allclose(residuals[valid], 0, atol=5e-7):
        diff = np.nanmax(np.abs(residuals[valid]))
        raise ValueError(f"Model did not match reference image: max abs diff = {diff}")
    else:
        logger.info("Model matches reference, yay!")
