#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocess_test.py - Make the required pixel, metadata, and PSF stores.
"""

import os, glob, sys
import time

import numpy as np
import argparse
from h5py import File
from astropy.io import fits

from forcepho.patches.storage import ImageNameSet, ImageSet, header_to_id
from forcepho.patches.storage import PixelStore, MetaStore


def find_images(loc=".", pattern="*noisy.fits"):
    import glob
    search = os.path.join(loc, pattern)
    files = glob.glob(search)
    names = [ImageNameSet(f,                        # im
                          f.replace("noisy", "unc"),  # err
                          "",  # mask
                          "")  # bkg
             for f in files]
    return names


def nameset_to_imset(nameset):
    # Read the header and set identifiers
    hdr = fits.getheader(nameset.im)
    band, expID = header_to_id(hdr, nameset)

    # Read data and perform basic operations
    # NOTE: we transpose to get a more familiar order where the x-axis
    # (NAXIS1) is the first dimension and y is the second dimension.
    im = np.array(fits.getdata(nameset.im)).T
    ierr = 1 / np.array(fits.getdata(nameset.err)).T
    imset = ImageSet(im=im, ierr=ierr, hdr=hdr,
                     band=band, expID=expID, names=nameset,
                     mask=None, bkg=None)
    return imset


def make_psf_store(psfstorefile, nradii=9, band="",
                   fwhm=[3.0], amp=[1.0],
                   data_dtype=np.float32):
    """Need to make an HDF5 file with <band>/psf datasets that have the form:
        psfs = np.zeros([nloc, nradii, ngauss], dtype=pdtype)
        pdtype = np.dtype([('gauss_params', np.float32, 6),
                           ('sersic_bin', np.int32)])
    and the order of `gauss_params` is given in patch.cu;
        amp, x, y, Cxx, Cyy, Cxy

    In this particular method we are using a single PSF for the entire
    image, and we are using the same number of gaussians (and same
    parameters) for each radius.

    Parameters
    ------------
    psfstorefile : string
        The full path to the file where the PSF data will be stored.
        Must not exist.

   nradii : int (default, 9)
        The number of copies of the PSF to include, corresponding to the number
        of sersic mixture radii.  This is because in principle each of the
        sersic mixture radii can have a separate PSF mixture.

    data_type : np.dtype
        A datatype for the gaussian parameters
    """
    cols = ["amp", "xcen", "ycen", "Cxx", "Cyy", "Cxy"]
    dtype = [(c, data_dtype) for c in cols] + [("sersic_bin", np.int32)]
    psf_dtype = np.dtype(dtype)

    with File(psfstorefile, "a") as h5:
        try:
            bg = h5.create_group(band)
        except(ValueError):
            del h5[band]
            bg = h5.create_group(band)

        ngauss = len(fwhm)
        amps = np.ones(ngauss) * np.array(amp)
        amps /= amps.sum()
        sigma = np.array(fwhm) / 2.355
        pars = np.zeros([1, nradii, ngauss], dtype=psf_dtype)
        bg.attrs["n_psf_per_source"] = int(nradii * ngauss)

        # Fill every radius with the parameters for the ngauss gaussians
        pars["amp"]     = amps
        pars["xcen"][:] = 0.0
        pars["ycen"][:] = 0.0
        pars["Cxx"]     = sigma**2
        pars["Cyy"]     = sigma**2
        pars["Cxy"]     = 0.0
        pars["sersic_bin"] = np.arange(nradii)[None, :, None]
        bg.create_dataset("parameters", data=pars.reshape(1, -1))
        #bg.create_dataset("detector_locations", data=pixel_grid)


if __name__ == "__main__":

    # --- Configuration ---
    # ---------------------

    # read config file
    from config_test import config

    # read command lines
    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    # --- Make pix and meta stores ---
    # --------------------------------
    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)
    # Make the (empty) metastore
    metastore = MetaStore()

    # --- Find Images ---
    names = find_images(loc=config.frames_directory,
                        pattern="*noisy.fits")

    # Fill pixel and metastores
    for n in names:
        ims = nameset_to_imset(n)
        pixelstore.add_exposure(ims)
        metastore.add_exposure(ims)

    bands = list(pixelstore.data.keys())

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)

    # --- PSF ---
    # -----------
    config.psf_fwhm = [hdr["PSFSIGMA"] / hdr["PIXSCALE"] * 2.355
                       for band in bands
                       for hdr in list(metastore.headers[band].values())[:1]]
    config.psf_amp = len(config.psf_fwhm) * [1.0]
    for band in config.bandlist:
        make_psf_store(config.psfstorefile, fwhm=config.psf_fwhm,
                       amp=config.psf_amp, band=band)
