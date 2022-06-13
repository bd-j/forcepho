#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

import matplotlib.animation as manimation
from astropy.io import fits

from forcepho.patches import FITSPatch, CPUPatchMixin
from forcepho.postprocess import Samples, Residuals

from pair_plot import plot_corner, plot_residual, plot_both, scatter


class Patcher(FITSPatch, CPUPatchMixin):
    pass


if __name__ == "__main__":

    # --- Arguments ---
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--patchname", type=str, default="output/v1/patches/patch1_samples.h5")
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # --- check out scene  & bounds ---
    samples = Samples(args.patchname)
    region, active, bounds = samples.region, samples.active, samples.bounds
    bands = samples.bands
    config = argparse.Namespace(**json.loads(samples.config))

    # --- load the image data ---
    patcher = Patcher(fitsfiles=[config.image_name],
                      psfstore=config.psfstorefile,
                      splinedata=config.splinedatafile,
                      return_residual=True)

    # --- prepare model and data, and sample ---
    deltas = []
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=None,
                                     bounds=bounds, shapes=samples.shape_cols)
    for q in samples.chain[::args.thin, :]:
        delta = model.residuals(q)[0].copy()
        deltas.append(delta)

    fig, raxes, paxes_all = plot_both(args.patchname, show_current=False)
    r = Residuals(args.patchname.replace("samples", "residuals"))
    data, _, _ = r.make_exp(value="data")
    delta, _, _ = r.make_exp(value="residual")
    ierr, _, _ = r.make_exp(value="ierr")
    paxes = np.array(paxes_all[:4]).reshape(2, 2)

    norm = raxes[0].get_images()[0].norm
    vmin, vmax = norm.vmin, norm.vmax
    kw = dict(origin="lower", vmin=vmin, vmax=vmax)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Pair Movie', artist='Matplotlib',
                    comment='Posterior draws for a pair of sources')
    writer = FFMpegWriter(fps=args.fps, metadata=metadata)
    
    # make sure output directory exists
    if not os.path.exists("movie/"):
        os.makedirs("movie/")

    with writer.saving(fig, "movie/pair_posterior.mp4", 200):

        for i, v in enumerate(samples.chaincat["CLEAR"][:, samples.n_tune::args.thin].T):
            delta = deltas[i].reshape(ierr.shape)
            raxes[1].clear()
            raxes[1].imshow((delta * ierr).T, **kw)
            raxes[1].set_title("Residual")
            raxes[2].clear()
            raxes[2].imshow(((data-delta) * ierr).T, **kw)
            raxes[2].set_title("Model")

            ymap = np.atleast_2d(v)
            scatter(ymap.T, paxes, zorder=20, color="grey", marker=".")
            for j, ax in enumerate(np.diag(paxes)):
                ax.axvline(ymap[0, j], color="grey", alpha=0.5)

            paxes_all[-1].axvline(ymap.sum(), color="grey", alpha=0.5)

            writer.grab_frame()
            #fig.savefig(f"movie/frame{i:03.0f}.png")
