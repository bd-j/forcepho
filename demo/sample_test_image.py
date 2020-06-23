#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import argparse
from astropy.io import fits
from astropy.wcs import WCS

from littlemcmc import _sample_one_chain as sample_one

from forcepho.proposal import Proposer
from forcepho.patches import JadesPatch
from forcepho.region import RectangularRegion
from forcepho.dispatcher import SuperScene, MPIQueue

from utils import Logger, rectify_catalog
from config_test import config

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


def prepare(patcher, active=None, fixed=None):
    """
    """
    # --- Build the things
    patcher.pack_meta(cat)
    gpu_patch = patcher.send_to_gpu()
    proposer = Proposer(patcher)
    proposer.patch.return_residual = True

    # --- Get parameter vector and proposal
    q = patcher.scene.get_all_source_params().copy()
    prop = patcher.scene.get_proposal()

    # --- subtract fixed sources ---
    if fixed is not None:
        q_fixed = q
        prop_fixed = prop
        out = proposer.evaluate_proposal(prop_fixed)
        residual_fixed = out[-1]

        patcher.pack_meta(active)
        q = patcher.scene.get_all_source_params().copy()
        prop = patcher.scene.get_proposal()
        patcher.swap_on_gpu()

    proposer.patch.return_residual = False

    return proposer, q


def bounds():
    pass


def make_model(proposer, q, **model_kwargs):
    """
    """
    ndim = len(q)
    lower, upper = bounds(proposer.patcher.scene)

    transform = BoundedTransform(lower, upper)
    model = GPUPosterior(proposer, proposer.patcher.scene,
                         transform=transform, upper=upper, lower=lower,
                         **model_kwargs)

    return model, q


def sample(model, q, draws, warmup=[10], z_cov=None):
    """
    """
    start = model.transform.inverse_transform(q)
    trace = None

    # --- Burn-in windows with step tuning ---
    for n_iterations in warmup:
        step = get_step_for_trace(init_cov=z_cov, trace=trace)
        trace, s = sample_one(logp_dlogp_func=model.lnprob_and_grad,
                              model_ndim=n_dim, start=start, step=step,
                              draws=n_iterations, tune=0,
                              discard_tuned_samples=False,
                              progressbar=False)
        start = trace[:, -1]

    # --- production run ---
    step = get_step_for_trace(init_cov=z_cov, trace=trace)
    trace, stats = sample_one(logp_dlogp_func=model.lnprob_and_grad,
                              model_ndim=n_dim, start=start, step=step,
                              draws=draws, tune=0,
                              discard_tuned_samples=True,
                              progressbar=True,)

    chain = model.transform.transform(trace)
    return chain, step, stats


def get_step(init_cov=None, N_init=0, trace=None):
    if trace is None and init_cov is None:
        cov = np.eye(model.ndim)
    elif trace is not None:
        cov = np.cov(trace, rowvar=False)
    else:
        cov = np.array(init_cov)

    potential = QuadPotentialFull(cov)
    return NUTS(potential=potential, **nuts_kwargs)


if __name__ == "__main__":

    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile)

    iexp = 0

    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    sceneDB = SuperScene(sourcecat=cat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch,
                         maxradius=config.patch_maxradius)

    while undone:
        region, active, fixed = sceneDB.checkout_region()

        # --- Build patch and fit ---
        patcher.build_patch(region, None, allbands=bands)
        proposer, q = prepare(patcher, active=active, fixed=fixed)
        model = make_model(proposer, q)
        chain, step, stats = sample(model, q, draws, warmup)

        # --- Deal with results ---
        qlast = chain[-1, :]
        patcher.scene.set_all_source_params(qlast)
        qcat = patcher.scene.to_catalog(extra_cols=["source_index"])
        qcat["source_index"][:] = active["source_index"]

        qcat, mass_matrix = dump(active, fixed, region, model, chain, step, stats)
        sceneDB.checkin_region(qcat, fixed, draws, mass_matrix=None)


    im = np.zeros([hdr["NAXIS1"], hdr["NAXIS2"], len(actives)])

    #sys.exit()

    for i, active in enumerate(actives):
        # TODO: This could work by successive data/residual swaps on the GPU
        result = get_model_gpu(active, patcher)
        model = data - result["residual"][0]
        im[xpix, ypix, i] += model

    image = im.sum(axis=-1).T
    with fits.HDUList(fits.PrimaryHDU(image)) as hdul:
        hdul[0].header.update(hdr[6:])
        hdul[0].header["NOISE"] = 0
        hdul.writeto(exp.replace("noisy", "force") + ".fits", overwrite=True)
