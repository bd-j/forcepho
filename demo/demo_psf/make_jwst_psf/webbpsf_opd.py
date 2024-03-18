#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import matplotlib.pyplot as pl

import webbpsf

charge_diff = webbpsf.constants.INSTRUMENT_DETECTOR_CHARGE_DIFFUSION_DEFAULT_PARAMETERS

JWST_BANDS = ["F070W", "F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W",
              "F182M", "F210M", "F430M", "F460M", "F480M",
              "F162M", "F250M", "F300M", "F150W2"]


#NAXIS1  =                  636
#NAXIS2  =                  636
#OVERSAMP=                    4 / Oversampling factor for FFTs in computation
#DET_SAMP=                    4 / Oversampling factor for MFT to detector plane
#PIXELSCL=              0.01575 / Scale in arcsec/pix (after oversampling)
#FOV     =               10.017 / Field of view in arcsec (full array)
#NWAVES  =                    9 / Number of wavelengths used in calculation


def trend(year, month):
    trend_table = webbpsf.trending.monthly_trending_plot(year, month, verbose=False)
    return trend_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--day", type=int, default=3)
    parser.add_argument("--jitter_sigma", type=float, default=0.001)
    parser.add_argument("--add_ipc", type=int, default=0,
                        help="IPC effects; if images are deconvolved, don't add this.")
    #parser.add_argument("--charge_diffusion_sigma", type=float, default=None)
    args = parser.parse_args()
    year, month, day = args.year, args.month, args.day
    outdir = f"./webbpsf-v{webbpsf.__version__}"

    tag = f"{year}-{month:02.0f}-{day:02.0f}"
    os.makedirs(f"{outdir}/{tag}_opd", exist_ok=True)
    oversamp, det_samp = 4, 4
    fov = 10.015 # arcsec
    detectors = dict(short="NRCA1", long="NRCA5")  # This is the default and was used for the ground computations.
    cd_nick = dict(short="NIRCAM_SW", long="NIRCAM_LW")
    nwave = dict(M=9, W=21)
    pos = (1024, 1024)
    table = trend(year, month)
    fig = pl.gcf()
    fig.savefig(f"{outdir}/{tag}_opd/{tag}_opd.png")
    iso_date = f"{year}-{month:02.0f}-{day:02.0f}T00:00:00"

    nrc = webbpsf.NIRCam()
    for band in JWST_BANDS:
        nrc.filter = band
        nrc.detector = detectors[nrc.channel]
        nick = cd_nick[nrc.channel]
        nrc.detector_position = pos
        nrc.load_wss_opd_by_date(iso_date, plot=False, choice="before")
        nrc.options['jitter'] = 'gaussian'
        nrc.options['jitter_sigma'] = args.jitter_sigma
        nrc.options['charge_diffusion_sigma'] = charge_diff[nick]
        nrc.options['add_ipc'] = bool(args.add_ipc)
        nrc.options['output_mode'] = 'both'

        psf_hdul = nrc.calc_psf(fov_arcsec=fov,
                                nlambda=nwave[band[-1]],
                                fft_oversample=oversamp,
                                detector_oversample=det_samp)
        for hdu in psf_hdul:
            hdu.header["CHRGDIFF"] = nrc.options['charge_diffusion_sigma']
            hdu.header["ADD_IPC"] = nrc.options['add_ipc']

        psf_hdul.writeto(f"{outdir}/{tag}_opd/PSF_NIRCam_{tag}_opd_filter_{band}.fits",
                         overwrite=True)