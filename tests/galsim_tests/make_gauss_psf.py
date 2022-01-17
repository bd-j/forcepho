#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from scipy.stats import multivariate_normal

from forcepho.patches.storage import PSF_COLS
from forcepho.mixtures.utils_hmc import Image

sigma_to_fwhm = 2.355

PSF_DTYPE = np.dtype([(c, np.float32) for c in PSF_COLS] +
                     [("sersic_bin", np.int32)])


def make_gauss_params(fwhm, nradii=9):

    sigma = np.atleast_1d(fwhm) / sigma_to_fwhm

    ngauss, nloc = len(sigma), 1
    pars = np.zeros([nloc, nradii, ngauss], dtype=PSF_DTYPE)
    pars["amp"] = 1.0
    pars["xcen"] = 0.0
    pars["ycen"] = 0.0
    pars["Cxx"] = (sigma**2)[None, None, :]
    pars["Cyy"] = (sigma**2)[None, None, :]
    pars["Cxy"] = 0.0
    pars["sersic_bin"] = np.arange(nradii)[None, :, None]

    return pars


def make_gauss_image(fwhm, det_samp=4):
    nx = int(np.ceil(fwhm * det_samp * 20))
    nx += int(np.mod(nx, 2))
    ny = nx
    x, y = np.meshgrid(np.arange(nx), np.arange(nx))
    sigma = fwhm * det_samp / sigma_to_fwhm
    cx, cy = nx / 2.0, ny / 2.0
    cov = np.eye(2) * sigma**2
    mvn = multivariate_normal([cx, cy], cov)
    data = mvn.pdf(np.dstack([x, y]))
    im = Image(x.flatten(), y.flatten(), data.flatten(), 0, nx, ny, cx, cy)
    return im


def package_output(h5file, band, pars, image, model=None,
                   scale=1, oversample=1, **header_kwargs):
    """
    """
    with h5py.File(h5file, "a") as h5:
        bg = h5.create_group(band.upper())
        bg.create_dataset("parameters", data=pars.reshape(pars.shape[0], -1))
        im = image.data.reshape(image.nx, image.ny)
        bg.create_dataset("truth", data=image.data)
        bg.create_dataset("xpix", data=image.xpix)
        bg.create_dataset("ypix", data=image.ypix)
        bg.attrs["n_psf_per_source"] = pars.shape[1] * pars.shape[2]
        bg.attrs["nx"] = image.nx
        bg.attrs["ny"] = image.ny
        bg.attrs["cx"] = image.cx
        bg.attrs["cy"] = image.cy
        bg.attrs["DET_SAMP"] = oversample
        bg.attrs["flux_scale"] = scale
        for k, v in header_kwargs.items():
            bg.attrs[k] = v

        if model is not None:
            m = model.reshape(image.nx, image.ny)
            bg.create_dataset("model", data=model)


if __name__ == "__main__":

    h5file = "data/stores/psf_gaussian_dec21.h5"
    fwhms = np.linspace(0.6, 3.0, 9)
    det_samps = np.floor(8 / np.ceil(fwhms))
    for i, fwhm in enumerate(fwhms):
        band = f"FWHM{fwhm:.1f}"
        det_samp = det_samps[i]
        pars = make_gauss_params(fwhm)
        im = make_gauss_image(fwhm, det_samp=det_samp)
        package_output(h5file, band, pars, im,
                       oversample=det_samp, FWHM=fwhm)