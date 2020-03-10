#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""storage.py

Interface with data on disk through storage objects.
"""

import os
from collections import namedtuple
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS

ImageNameSet = namedtuple("Image", ["im", "err", "mask", "bkg"])

EXP_FMT = "{}/{}"

# should match order in patch.cu
PSF_COLS = ["amp", "xcen", "ycen", "Cxx", "Cyy", "Cxy"]


def header_to_id(hdr, nameset):
    band = hdr["FILTER"]
    expID = os.path.basename(nameset.im).replace(".fits", "")
    return band, expID


class PixelStore:
    """Organization of the pixel data store is

    `bandID/expID/data`

    where `data` is an array of shape (nsuper, nsuper, 2*super_pixel_size**2) 
    The first half of the trailing dimension is the pixel flux information,
    while the second half is the ierr information. Each dataset has attributes
    that describe the nominal flux calibration that was applied and some
    information about the subtracted background, mask, etc.

    Parameters
    ----------
    nside_full : int or 2-element array of ints, optional (default: 2048)
        The number of pixels along each dimension of the image.  Must be an
        integer multiple of the `super_pixel_size`

    super_pixel_size : int, optional (default: 8)
        The number of pixels along each side of a super-pixel

    pix_dtype : numpy.dtype, optional (default: np.float32)
        The data type of the pixels.
    """

    def __init__(self, h5file, nside_full=2048, super_pixel_size=8,
                 pix_dtype=np.float32):

        self.h5file = h5file
        try:
            # file and attrs exist, use stored value
            with h5py.File(self.h5file, "r") as h5:
                self.nside_full = h5.attrs["nside_full"]
                self.super_pixel_size = h5.attrs["super_pixel_size"]
        except(KeyError, OSError):
            # file or attrs do not exist, create them
            nside_full = np.array(nside_full)
            print("Adding nside_full={}, super_ixel_sze={} to attrs".format(nside_full, super_pixel_size))
            #super_pixel_size = np.array(super_pixel_size)
            with h5py.File(self.h5file, "a") as h5:
                h5.attrs["nside_full"] = nside_full
                h5.attrs["super_pixel_size"] = super_pixel_size
            self.nside_full = nside_full
            self.super_pixel_size = super_pixel_size

        self.pix_dtype = pix_dtype
        self.nside_super = self.nside_full / self.super_pixel_size
        self.xpix, self.ypix = self.pixel_coordinates()

    def superpixel_corners(self, imsize=None):
        """
        Returns
        ---------
        corners : ndarray of shape (nside_super, nside_super, 4, 2)
            The coordinates in the full array of the corner pixels of each of
            the superpixels.
        """
        if not imsize:
            xpix, ypix = self.xpix, self.ypix
        else:
            # This is inefficient
            xpix, ypix = self.pixel_coordinates(imsize=imsize)
        # Full image coordinates of the super pixel corners
        lower_left = np.array([xpix[:, :, 0], ypix[:, :, 0]])
        offsets = np.array([(0, 0), (1, 0), (1, 1), (0, 1)]) * (self.super_pixel_size - 1)
        corners = offsets[:, :, None, None] + lower_left[None, :, :, :]

        #xx = xpix[:, :, 0], xpix[:, :, -1]
        #yy = ypix[:, :, 0], ypix[:, :, -1]
        # corners = np.array([(xx[0], yy[0]), (xx[1], yy[0]), (xx[1], yy[1]), (xx[0], yy[1])])

        return corners.transpose(2, 3, 0, 1)

    def pixel_coordinates(self, imsize=None):
        if not imsize:
            imsize = np.zeros(2) + self.nside_full
        # NOTE: the order swap here for x, y
        yy, xx = np.meshgrid(np.arange(imsize[1]), np.arange(imsize[0]))
        packed = self.superpixelize(xx, yy)
        xpix = packed[:, :, :self.super_pixel_size**2].astype(np.int16)
        ypix = packed[:, :, self.super_pixel_size**2:].astype(np.int16)
        return xpix, ypix

    def add_exposure(self, nameset, bitmask=None):
        """Add an exposure to the pixel data store, including background
        subtraction (if `nameset.bkg`), flux conversion, setting ierr for masked
        pixels to 0, and super-pixel ordering.  This opens the HDF5 files, adds
        the image data, and closes the file.

        Parameters
        -------------
        nameset : NamedTuple with attributes `im`, `err`, `bkg`, `mask`
            A set of names (including path) for a given exposure.  Values of
            `None` or `False` for bkg and mask will result in no background
            subtraction and no masking beyond NaNs and infs
        """
        # Read the header and set identifiers
        hdr = fits.getheader(nameset.im)
        band, expID = header_to_id(hdr, nameset)

        # Read data and perform basic operations
        # NOTE: we transpose to get a more familiar order where the x-axis
        # (NAXIS1) is the first dimension and y is the second dimension.
        im = np.array(fits.getdata(nameset.im)).T
        ierr = 1 / np.array(fits.getdata(nameset.err)).T
        if nameset.bkg:
            bkg = np.array(fits.getdata(nameset.bkg)).T
            im -= bkg
        else:
            bkg = np.zeros_like(im)
        mask = ~(np.isfinite(ierr) & np.isfinite(im) & (ierr >= 0))
        if nameset.mask:
            pmask = np.array(fits.getdata(nameset.mask)).T
            if bitmask:
                # just check that any of the bad bits are set
                pmask = np.bitwise_and(pmask, bitmask) != 0
            mask = mask | pmask
        ierr[mask] = 0
        # masked pixels provide a sampling of the subtracted background
        im[mask] = bkg[mask]
        # this does nominal flux calibration of the image.
        # Returns the calibration factor applied
        fluxconv, unitname = self.flux_calibration(hdr)
        im *= fluxconv
        ierr *= 1. / fluxconv

        # Superpixelize
        imsize = np.array(im.shape)
        assert np.all(np.mod(imsize, self.super_pixel_size) == 0)
        if np.any(imsize != self.nside_full):
            # In principle this can be handled, but for now we assume all
            # images are the same size
            raise ValueError("Image is not the expected size")
        superpixels = self.superpixelize(im, ierr)

        # Put into the HDF5 file; note this opens and closes the file
        with h5py.File(self.h5file, "r+") as h5:
            path = "{}/{}".format(band, expID)
            try:
                exp = h5.create_group(path)
            except(ValueError):
                del h5[path]
                print("deleted existing data for {}".format(path))
                exp = h5.create_group(path)
            pdat = exp.create_dataset("data", data=superpixels)
            pdat.attrs["counts_to_flux"] = fluxconv
            pdat.attrs["flux_units"] = unitname
            if bitmask:
                pdat.attrs["bitmask_applied"] = bitmask
            for i, f in enumerate(nameset._fields):
                if type(nameset[i]) is str:
                    pdat.attrs[f] = nameset[i]

    def superpixelize(self, im, ierr, pix_dtype=None):
        """Take native image data and reshape into super-pixel order
        """
        super_pixel_size = self.super_pixel_size
        s2 = super_pixel_size**2
        nsuper = (np.array(im.shape) / super_pixel_size).astype(int)
        if not pix_dtype:
            pix_dtype = self.pix_dtype
        superpixels = np.empty([nsuper[0], nsuper[1], 2 * super_pixel_size**2],
                               dtype=pix_dtype)
        # slow, could be done with a reshape...
        for ii in range(nsuper[0]):
            for jj in range(nsuper[1]):
                I = ii * super_pixel_size
                J = jj * super_pixel_size
                superpixels[ii, jj, :s2] = im[I:(I + super_pixel_size),
                                              J:(J + super_pixel_size)].flatten()
                superpixels[ii, jj, s2:] = ierr[I:(I + super_pixel_size),
                                                J:(J + super_pixel_size)].flatten()
        return superpixels

    def flux_calibration(self, hdr):
        if "ABMAG" in hdr:
            zp = hdr["ABMAG"]
            image_units = "nJy"
            # math from Sandro
            conv = 1e9 * 10**(0.4 * (8.9 - zp))
        else:
            print("Warning, no phootmetric calibration applied")
            image_units = "counts"
            conv = 1.0
        return conv, image_units

    # Need better file handle treatment here.
    # should test for open file handle and return it, otherwise create and cache it
    @property
    def data(self):
        return h5py.File(self.h5file, "a", swmr=True)


class MetaStore:
    """Storage for exposure meta data.

         * headers - dictionary of FITS headers, keyed by band and then expID
         * wcs - dictionary of wcs objects, keyed by band and expID
    """
    def __init__(self, metastorefile=None):
        if not metastorefile:
            self.headers = {}
        else:
            self.headers = self.read_from_file(metastorefile)
            self.populate_wcs()

    def populate_wcs(self):
        self.wcs = {}
        for band in self.headers.keys():
            self.wcs[band] = {}
            for expID, hdr in self.headers[band].items():
                w = WCS(hdr)
                # remove extraneous axes
                if w.naxis == 3:
                    w = w.dropaxis(2)
                self.wcs[band][expID] = w

    def add_exposure(self, nameset):
        hdr = fits.getheader(nameset.im)
        band, expID = header_to_id(hdr, nameset)
        if band not in self.headers:
            self.headers[band] = {}
        self.headers[band][expID] = hdr

    def write_to_file(self, filename):
        """Convert the FITS headers in the dictionary to strings, and dump the
        dictionary to a file using JSON.

        Parameters
        ----------
        filename : str
            The name of the file fro the metadata.  Will be overwritten if it
            already exists
        """
        import json
        hstrings = {}
        for band in self.headers.keys():
            hstrings[band] = {}
            for expID, hdr in list(self.headers[band].items()):
                hstrings[band][expID] = hdr.tostring()
        with open(filename, "w") as f:
            json.dump(hstrings, f)

    def read_from_file(self, filename):
        """Read a json serialized dictionary of string headers
        """
        H = fits.Header()
        import json
        headers = {}
        with open(filename, "r") as f:
            sheaders = json.load(f)
        for band in sheaders.keys():
            headers[band] = {}
            for expID, h in sheaders[band].items():
                headers[band][expID] = H.fromstring(h)

        return headers


class PSFStore:
    """Assumes existence of a file with the following structure

    band/detector_locations
    band/psfs

    where psfs is a dataset like
      psfs = np.zeros(nloc, nradii, ngauss, dtype=pdt)
      pdt = np.dtype([('gauss_params', np.float, 6),
                      ('sersic_bin', np.int32)])
    and the order of gauss_params is given in patch.cu; amp, x, y, Cxx, Cyy, Cxy

    In principle ngauss can depend on i_radius
    """

    def __init__(self, h5file):
        self.h5file = h5file

    def lookup(self, band, xy=None):
        """Returns a array of shape (nradii x ngauss,) with dtype
        """
        try:
            x, y = xy
            xp, yp = self.data[band]["detector_locations"][:]
            dist = np.hypot(x - xp, y - yp)
            choose = dist.argmin()
        except:
            choose = 0
        pars = self.data[band]["parameters"][choose]
        # TODO: assert data dtype is what's required for JadesPatch
        #assert pars.dtype.descr
        return pars

    def get_local_psf(self, band="F090W", source=None, wcs=None):
        """
        Returns
        --------
        A structured array of psf parameters for a given source in a give band.
        The structure of the array is something like
        (amp, xcen, ycen, Cxx, Cyy Cxy, sersic_radius_index)
        There are npsf_per_source rows in this array.
        """
        if wcs:
            xy = wcs.all_world2pix(source.ra, source.dec)
        else:
            xy = None
        psf = self.lookup(band, xy=xy)

        return psf

    @property
    def data(self):
        return h5py.File(self.h5file, "r")