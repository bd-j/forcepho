#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, os, sys
import argparse
import numpy as np

from astropy.io import fits
import stwcs
from drizzlepac import adrizzle

from demo_utils import write_fits_to


def drizzle_combine(ditherlist, args, wt_scale=1e-4):
    """
    Parameters
    ----------
    ditherlist : list of strings
        The paths to the individual exposures.  It is assumed that each exposure
        is a FITS image with the following data model:
        * EXT0 - flux image
        * EXT1 - rms image
        * EXT[-1] - scene catalog
    """

    with fits.open(ditherlist[0], memmap=False) as hdul:
        scene = np.array(hdul[-1].data)
        bands = hdul[-1].header["FILTERS"].split(",")
        hdr_out = hdul[0].header
        wcs_out = stwcs.wcsutil.HSTWCS(hdul, wcskey=' ')

    sci_final = np.zeros(wcs_out.array_shape)
    wht_final = np.zeros(wcs_out.array_shape)

    for exp in ditherlist:
        # get the input
        with fits.open(exp) as hdul:
            sci = hdul["SCI"].data.astype(np.float32)
            wht = 1 / (hdul["ERR"].data)**2  # inverse variance
            hdr = hdul["SCI"].header
            wcs_in = stwcs.wcsutil.HSTWCS(hdul, wcskey=' ')

        #wht = wht / np.median(wht)
        #wht_in = (np.sqrt(wht) * wt_scale).astype(np.float32)
        #wht_in = np.ones_like(sci)
        wht_in = (wht * wt_scale).astype(np.float32)

        # set up the output
        wht_out = np.zeros(wcs_out.array_shape, dtype=np.float32)
        sci_out = np.empty(wcs_out.array_shape, dtype=np.float32) * np.nan
        outcon = np.zeros((1,) + wcs_out.array_shape, dtype=np.int32)

        # drizzle
        adrizzle.do_driz(sci, wcs_in, wht_in,
                         wcs_out, sci_out, wht_out, outcon,
                         0.0, "cps", 1.0, wcslin_pscale=args.scale,
                         pixfrac=args.pixfrac, kernel="square")

        # write the drizzled single file output
        out = exp.replace("dither", "drz_dither")
        write_fits_to(out, sci_out.T, (1 / np.sqrt(wht_out/wt_scale)).T, hdr_out)

        # accumulate in the final image
        w_single = wht_out / wt_scale
        wprime = wht_final + w_single
        sci_final = (sci_final * wht_final + sci_out * w_single) / wprime
        wht_final = wprime
        print(np.nanmax(sci_final))

    # write the mosaic
    out = os.path.join(os.path.dirname(exp), "mosaic.fits")
    write_fits_to(out, sci_final.T, 1 / np.sqrt(wht_final.T), hdr_out, bands, scene=scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--pixfrac", type=float, default=0.8)
    parser.add_argument("--scale", type=float, default=0.03)
    args = parser.parse_args()

    ditherlist = glob.glob(os.path.join(args.outdir, "dither_0*fits"))

    mosaic_name = drizzle_combine(ditherlist, args)
