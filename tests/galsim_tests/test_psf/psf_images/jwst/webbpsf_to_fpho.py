#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits


JWST_BANDS = ["F070W", "F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W",
              "F182M", "F210M", "F430M", "F460M", "F480M"]


if __name__ == "__main__":
    bands = JWST_BANDS
    ext = 0  # PSF oversampled by 4x relative to detector sampling.
    for band in bands:
        imname = f"PSF_NIRCam_predicted_opd_filter_{band}.fits"
        with fits.open(imname) as hdul:
            im = hdul[ext].data
            hdr = hdul[ext].header
        fits.writeto(f"{band.lower()}_psf.fits", im, hdr)