#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import h5py

from .region import CircularRegion
from .utils import read_config, make_chaincat, make_statscat
from .fitting import Result
from .patches import JadesPatch
from .proposal import Proposer


__all__ = ["Samples", "Reconstructor",
           "show_exp"]


def _make_imset(out, paths, name, arrs):
    for i, epath in enumerate(paths):
        try:
            g = out[epath]
        except(KeyError):
            g = out.create_group(epath)

        try:
            g.create_dataset(name, data=np.array(arrs[i]))
        except:
            print("Could not make {}/{} dataset from {}".format(epath, name, arrs[i]))


class Samples(Result):

    def __init__(self, filename):
        self.read_patch(filename)

    def read_patch(self, filename):

        self.read_from_h5(filename)
        self.region = CircularRegion(self.ra, self.dec, self.radius)
        self.bands = [b.decode("utf") for b in self.bandlist]
        self.shape_cols = [s.decode("utf") for s in self.shapenames]

        self.chaincat = make_chaincat(self.chain, self.bands, self.active,
                                      self.reference_coordinates, shapes=self.shape_cols)
        self.n_active = len(self.active)
        self.dtype_sample = np.dtype([desc[:2] for desc in self.chaincat.dtype.descr])

    def get_sample_cat(self, iteration):
        sample = np.zeros(self.n_active, dtype=self.dtype_sample)
        for d in sample.dtype.names:
            try:
                sample[d] = self.chaincat[d][:, iteration]
            except(IndexError):
                sample[d] = self.chaincat[d]
        return sample

    def show_chain(self, source_idx=0, bandlist=None, axes=None,
                   span=0.999999426697, post_kwargs=dict(alpha=0.5, color="royalblue"),
                   truth=None, truth_kwargs=dict(linestyle="--", color="tomato")):
        if bandlist is None:
            bandlist = self.bands
        cols = bandlist + self.shape_cols
        q = 100 * np.array([0.5 - 0.5 * span, 0.5 + 0.5 * span])
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            xx = self.chaincat[source_idx][col]
            lim = np.percentile(xx, list(q))
            if self.bounds is not None:
                lim = self.bounds[source_idx][col]

            ax.plot(xx, **post_kwargs)
            ax.set_ylim(*lim)
            ax.set_ylabel(col)
            if truth is not None:
                ax.axhline(truth[col], **truth_kwargs)
        return ax


class Reconstructor:

    def __init__(self, patcher, MAXSOURCES=15):

        self.pixattrs = ["xpix", "ypix", "ierr", "original", "residual"]
        self.metas = ["D", "CW", "crpix", "crval"]

        self.patcher = patcher
        self.MAXSOURCES = MAXSOURCES

    def fetch_data(self, region, bands):
        # --- get pixel data and metadata---
        self.patcher.build_patch(region, None, allbands=bands)
        self.original = self.patcher.split_pix("data")
        self.xpix = self.patcher.split_pix("xpix")
        self.ypix = self.patcher.split_pix("ypix")
        self.ierr = self.patcher.split_pix("ierr")
        self.epaths = self.patcher.epaths
        self.bands = bands

    def model_data(self, parameters, split=True):
        """Model the pixels for a given scene
        """
        inds = np.arange(self.MAXSOURCES, len(parameters), self.MAXSOURCES)
        blocks = np.array_split(parameters, inds)

        model = np.zeros_like(self.patcher.data)
        for i, block in enumerate(blocks):
            residual = self.get_residuals(block, split=False)
            model += self.patcher.data - residual

        if split:
            model = np.split(model, np.cumsum(self.patcher.exposure_N)[:-1])
        return model

    def get_residuals(self, parameters, split=True):
        """Get the pixel residuals for a given set of source parameters.
        The number of sources must fit on the GPU
        """
        # --- pack parameters and send to gpu ---
        self.patcher.pack_meta(parameters)
        gpu_patch = self.patcher.send_to_gpu()
        proposer = Proposer(self.patcher)
        proposal = self.patcher.scene.get_proposal()
        q = self.patcher.scene.get_all_source_params().copy()

        # --- evaluate proposal, including residuals
        self.patcher.return_residual = True
        out = proposer.evaluate_proposal(proposal)
        residual = out[-1]
        if split:
            residual = np.split(residual, np.cumsum(self.patcher.exposure_N)[:-1])

        return residual

    def build_arrays(self, results, parameters=None):
        """Build the relevant arrays based on a Samples() structure instance.
        Only works if len(parameters) <= self.MAXSOURCES
        """
        self.results = results
        self.fixed = results.fixed
        self.reference_coordinates = results.reference_coordinates
        self.shapes = results.shape_cols
        self.parameters = parameters
        if parameters is None:
            self.parameters = results.get_sample_cat(-1)

        assert len(parameters) <= self.MAXSOURCES

        # get the pixels
        self.fetch_data(results.region, results.bands)

        # model the pixels for these parameters, and cache the meta parameters
        residual = self.get_residuals(self.parameters, split=False)
        for a in self.metas:
            setattr(self, a, getattr(self.patcher, a))
        if self.fixed:
            rf = self.get_residuals(self.fixed, split=False)
            residual += rf - self.patcher.data

        self.residual = np.split(residual, np.cumsum(self.patcher.exposure_N)[:-1])

    def write_arrays(self, filename):
        """Write relevant arrays to HDF5 file
        """
        with h5py.File(filename, "w") as out:
            out.create_dataset("epaths", data=np.array(self.epaths, dtype="S"))
            out.create_dataset("exposure_start", data=self.patcher.exposure_start)
            out.create_dataset("active", data=self.parameters)
            out.create_dataset("fixed", data=self.results.fixed)

        for band in self.bands:
            g = out.create_group(band)

        for a in self.pixattrs + self.metas:
            arr = getattr(self, a)
            _make_imset(out, self.epaths, a, arr)

    def read_arrays(self, filename):
        """Read relevant arrays from HDF5 file
        """
        attrs = self.pixattrs + self.metas
        [setattr(self, a, []) for a in attrs]
        with h5py.File(filename, "r") as out:
            self.epaths = out["epaths"][:]
            self.parameters = out["parameters"][:]
            self.fixed = out["fixed"][:]
            for i, e in enumerate(self.epaths):
                for a in attrs:
                    setattr(self, a, getattr(self, a).append(out[e][a][:]))

    def get_model_image(self, model, iexp=0, epath=""):
        """Generate the model and use it to fill pixels corresponding to the original image
        """
        if not epath:
            epath = self.patcher.epaths[iexp]
        hdr = self.patcher.metastore.headers[epath]
        image = np.zeros([hdr["NPIX1"], hdr["NPIX2"]])
        image[self.xpix[iexp], self.ypix[iexp]] += model[iexp]
        return image, hdr, epath

    def show_residuals(self, figure_name=None, exposure_inds=[0, -1],
                       show_fixed=True, show_active=True, imshow_kwargs={}):

        active = self.parameters
        fixed = self.fixed
        ref = self.reference_coordinates
        nexp = len(exposure_inds)

        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(nexp, 3, sharex="row", sharey="row", squeeze=False)

        for i, e in enumerate(exposure_inds):
            ee = self.epaths[e].decode("utf")
            model = self.original[e] - self.residual[e]
            pix = [self.original[e], self.residual[e], model]
            x, y = self.xpix[e], self.ypix[e]

            for j, v in enumerate(pix):
                ax = axes[i, j]
                show_exp(x, y, v, ax=ax, **imshow_kwargs)
            axes[i, 0].set_ylabel(" ".join(ee.replace(".flx", "").split("_")[-3:]))

            if show_active:
                ax = self.mark_sources(active["ra"], active["dec"], e,
                                       ref_coords=ref, ax=ax, color="red")
            if show_fixed:
                for j in range(3):
                    ax = axes[i, j]
                    ax = self.mark_sources(fixed["ra"], fixed["dec"], e,
                                           ref_coords=ref, ax=ax, color="magenta")
            if figure_name:
                fig.savefig(figure_name)

            return fig, axes

    def mark_sources(self, ra, dec, exp_idx, ref_coords=None, ax=None,
                     plot_kwargs={"marker": "x", "linestyle": "", "color": "red"},
                     **extras):

        pix = self.sky_to_pix(ra, dec, exp_idx, ref_coords=ref_coords)

        plot_kwargs.update(extras)
        ax.plot(pix[:, 0], pix[:, 1], **plot_kwargs)
        return ax

    def sky_to_pix(self, ra, dec, exp_idx=0, ref_coords=0.):

        e = exp_idx
        # honestly this should query the full WCS using
        # get_local_linear for each ra,dec pair
        crval = self.crval[e]
        crpix = self.crpix[e]
        CW = self.CW[e]

        i = 0
        if len(CW) != len(ra):
            CW = CW[i]
            crval = crval[i]
            crpix = crpix[i]

        sky = np.array([ra, dec]).T - (crval + ref_coords)
        pix = np.matmul(CW, sky[:, :, None])[..., 0] + crpix

        return pix


def show_exp(xpix, ypix, value, ax=None, **imshow_kwargs):
    """Create a rectangular image that bounds the given pixel coordinates
    and assign `value` to the correct pixels. Pixels in the rectangle that do
    not have assigned values are given nan.  use imshow to display the image in
    standard astro format (x increasing left to right, y increasing bottom to
    top)
    """
    lo = np.array((xpix.min(), ypix.min())) - 0.5
    hi = np.array((xpix.max(), ypix.max())) + 0.5
    size = hi - lo
    im = np.zeros(size.astype(int)) + np.nan

    x = (xpix-lo[0]).astype(int)
    y = (ypix-lo[1]).astype(int)
    # This is the correct ordering of xpix, ypix subscripts
    im[x, y] = value

    ax.imshow(im.T, origin="lower",
              extent=(lo[0], hi[0], lo[1], hi[1]),
              **imshow_kwargs)
    return ax



if __name__ == "__main__":

    # --- Config ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="galsim.yml")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--result_file", type=str, default="./output/smoketest/")
    parser.add_argument("--residuals_file", type=str, default="./output/smoketest/patches/")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config_file, args)


    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)

    recon = Reconstructor(patcher)
    result = Samples(config.result_file)

    # make and save residuals
    if config.write:
        recon.build_arrays(result, result.chaincat[-1])
        recon.write_arrays(config.residuals_file)
    # or read residuals
    else:
        recon.read_arrays(config.residuals_file)

    recon.show_residuals(figure_name=None)