#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, glob
from astropy.io import fits


JWST_BANDS = ["F070W", "F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W",
              "F182M", "F210M", "F430M", "F460M", "F480M"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="./")
    parser.add_argument("--outdir", type=str, default="./")
    parser.add_argument("--bands", type=str, nargs="*", default=JWST_BANDS)
    args = parser.parse_args()
    bands = args.bands
    ext = 2  # PSF oversampled by 4x relative to detector sampling, including physical blurring and distortion
    for band in bands:
        imname = os.path.join(args.indir, f"PSF_NIRCam_*_opd_filter_{band.upper()}.fits")
        imname = glob.glob(imname)[0]
        with fits.open(imname) as hdul:
            im = hdul[ext].data
            hdr = hdul[ext].header
        outname = os.path.join(args.outdir, f"{band.lower()}_psf.fits")
        fits.writeto(outname, im, hdr, overwrite=True)