# -*- coding: utf-8 -*-

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from .storage import MetaStore, PixelStore
from .patch import Patch


__all__ = ["PixelPatch",
           "StorePatch", "FITSPatch",
           "JWST_BANDS"]


JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]


# FIXME: Better logic for background offsets

class PixelPatch(Patch):

    """This class provides a method for packing pixel data in the correct format
    and keeping track of bookkeeping numbers, given lists of exposure locations,
    wcses, and a region.  Subclasses must implement :py:meth:`build_patch` and
    :py:meth:`find_pixels`
    """

    def __init__(self,
                 psfstore="",
                 splinedata="",
                 spline_smoothing=None,
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 debug=0,
                 ):

        super().__init__(return_residual=return_residual,
                         meta_dtype=meta_dtype,
                         pix_dtype=pix_dtype,
                         debug=debug,
                         psfstore=psfstore,
                         splinedata=splinedata,
                         spline_smoothing=spline_smoothing)

        self.background_offsets = None
        self.max_snr = None

    def build_patch(self, region=None, sourcecat=None,
                    allbands=JWST_BANDS, tweak_background=False):
        """Given a region this method finds and packs up all the relevant
        pixel-data in a format suitable for transfer to the GPU.  Optionally
        pack up metadata if a source catalog is provided.

        This fills the following attributes:

        * epaths - list of valid image names
        * `hdrs` - list of FITS headers, matches `epaths`
        * `wcses` - list of WCS objects, matches `epaths`
        * `bands` - list of filter names, matches `epaths`
        * `bandlist` - list of unique filter names
        * `n_bands` - number of unique bands
        * `n_exp` - number of valid individual eposures

        Parameters
        ---------
        region : forcepho.region.Region()
            An instance of a `Region` object used to find relevant exposures
            and pixels for the desired patch

        sourcecat : structured array
            A structured array describning the parameters of the sources in
            the scene.  The relevant columns are given by `storage.PAR_COLS`

        allbands : list of strings (optional)
            The names of the bands in the `flux` column of the source cat,
            corresponding to the "FILTER" keyword of FITS image headers or keys
            of the pixel and meta stores

        tweak_background : str, optional (default, empty string)
            If given, collect exposure dependent backgrounds stored in the
            the metadata header key given by this string.  If this header key
            is not present, uses a value of 0.0.  These backgrounds are
            subtracted during pixel packing.
        """
        # --- Find relevant exposures ---
        # The output lists are all of length n_exp and should all be in band
        # order
        meta = self.find_exposures(region, allbands)
        self.hdrs, self.wcses, self.epaths, self.bands = meta
        if len(self.epaths) == 0:
            raise ValueError("No exposures in the specified bands overlap the region")

        if tweak_background:
            self.background_offsets = [hdr.get(tweak_background, 0.0)
                                       for hdr in self.hdrs]
        else:
            self.background_offsets = None

        # --- Get BAND information for the exposures ---
        # band_ids must be an int identifier (does not need to be contiguous)
        allbands = list(allbands)
        band_ids = [allbands.index(b) for b in self.bands]
        assert (np.diff(band_ids) >= 0).all(), 'Exposures must be sorted by band'
        u, n = np.unique(band_ids, return_counts=True)
        self.uniq_bands, self.n_exp_per_band = u, n
        self.bandlist = [allbands[i] for i in self.uniq_bands]

        # --- Cache some useful numbers ---
        self.n_bands = len(self.uniq_bands)       # Number of bands/filters
        self.n_exp = len(self.hdrs)               # Number of exposures

        # --- Pack up all the data for the gpu ---
        self.pack_pix(region=region)
        if sourcecat is not None:
            scene = self.set_scene(sourcecat)
            self.pack_meta(scene)

        self._dirty_data = False

    def pack_pix(self, region, dtype=None):
        """We have super-pixel data in individual exposures that we want to
        pack into concatenated 1D pixel arrays.

        As we go, we want to build up the index arrays that allow us to find
        an exposure in the 1D arrays.

        Fills the following arrays:
        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start     [NBAND] exposure index corresponding to the start of each band
        - self.band_N         [NBAND] number of exposures in each band
        - self.exposure_start [NEXP]  pixel index corresponding to the start of each exposure
        - self.exposure_N     [NEXP]  number of pixels (including warp padding) in each exposure
        """
        if not dtype:
            dtype = self.pix_dtype

        # These index the exposures
        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)
        self.band_N_pix = np.empty(self.n_bands, dtype=np.int32)

        # wish I knew how many superpixels there would be;
        # One could set an upper bound based on region area * nexp
        #total_padded_size = region.area * n_exp
        #self.xpix = np.zeros(total_padded_size, dtype=dtype)
        data, ierr, xpix, ypix = [], [], [], []

        b, i = 0, 0
        for e, wcs in enumerate(self.wcses):
            # Get pixel data from this exposure;
            # NOTE: these are in super-pixel order
            pixdat = self.find_pixels(self.epaths[e], wcs, region)
            # use size instead of len here because we are going to flatten.
            n_pix = pixdat[0].size
            msg = f"There were no valid pixels in exposure {self.epaths[e]}"
            assert n_pix > 0, msg
            if self.debug > 0:
                msg = f"There were non-finite pixels in exposure {self.epaths[e]}"
                assert np.all(np.isfinite(pixdat[0] * pixdat[1])), msg

            # HACK: this could be cleaned up
            if self.background_offsets is not None:
                fluxes = pixdat[0] - self.background_offsets[e]
            else:
                fluxes = pixdat[0]

            if e > 0 and self.bands[e] != self.bands[e - 1]:
                b += 1
            self.band_N[b] += 1
            self.exposure_start[e] = i
            self.exposure_N[e] = n_pix
            data.append(fluxes)
            ierr.append(pixdat[1])
            xpix.append(pixdat[2])
            ypix.append(pixdat[3])
            i += n_pix

        #print(i)
        # Flatten and set the dtype explicitly here
        self.data = np.concatenate(data).reshape(-1).astype(dtype)
        assert self.data.shape[0] == i, "pixel data array is not the right shape"
        self.ierr = np.concatenate(ierr).reshape(-1).astype(dtype)
        self.xpix = np.concatenate(xpix).reshape(-1).astype(dtype)
        self.ypix = np.concatenate(ypix).reshape(-1).astype(dtype)
        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]
        self.band_N_pix[:-1] = np.diff(self.exposure_start[self.band_start])
        self.band_N_pix[-1] = self.npix - self.band_N_pix[:-1].sum()

        if getattr(self, "max_snr", None):
            self._cap_snr(max_snr=self.max_snr)

    def _cap_snr(self, max_snr=None):
        if max_snr:
            to_cap = (self.ierr > 0) & (self.data * self.ierr > max_snr)
            self.ierr[to_cap] = max_snr / self.data[to_cap]

    def find_exposures(self, region, bandlist):
        raise NotImplementedError

    def find_pixels(self, epath, wcs, region):
        raise NotImplementedError


class StorePatch(PixelPatch):

    """This class converts between JADES-like exposure level pixel data,
    meta-data (WCS), and PSF information stored in HDF5 pixel and meta stores to
    the data formats required by the device-side code.

    Parameters
    ----------
    pixelstore : string
        Path to HDF5 file to be used for
        :py:class:`forcepho.patches.storage.PixelStore`

    metastore : string
        Path to json file containing associated metadata for the pixel store.
        Used to instantiate :py:class:`forcepho.patches.storage.MetaStore`.
   """

    def __init__(self,
                 pixelstore="",
                 metastore="",
                 psfstore="",
                 splinedata="",
                 spline_smoothing=None,
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 debug=0,
                 ):

        super().__init__(return_residual=return_residual,
                         meta_dtype=meta_dtype,
                         pix_dtype=pix_dtype,
                         debug=debug,
                         psfstore=psfstore,
                         splinedata=splinedata,
                         spline_smoothing=spline_smoothing)

        self.metastore = MetaStore(metastore)
        self.pixelstore = PixelStore(pixelstore)

    def find_exposures(self, region, bandlist):
        """Return a list of headers (dict-like objects of wcs, filter, and
        exposure id) and exposureIDs for all exposures that overlap the region.
        These should be sorted by integer band_id.

        Parameters
        ----------
        region : region.Region instance
            Exposures will be found which overlap this region

        bandlist : list of str
            A list of band names to search for images.

        Returns
        -------
        hdrs, wcses, epaths, bands
        """
        imsize = np.zeros(2) + self.pixelstore.nside_full
        super_corners = self.pixelstore.superpixel_corners()
        bra, bdec = region.bounding_box

        epaths, bands, hdrs, wcses = [], [], [], []
        for band in bandlist:
            if band not in self.metastore.wcs.keys():
                continue
            for expID in self.metastore.wcs[band].keys():
                epath = "{}/{}".format(band, expID)
                wcs = self.metastore.wcs[band][expID]
                # Check region bounding box has a corner in the exposure.
                # NOTE: If bounding box entirely contains image this might fail
                bx, by = wcs.world_to_pixel_values(bra, bdec)
                inim = np.any((bx > 0) & (bx < imsize[0]) &
                              (by > 0) & (by < imsize[1]))
                if inim:
                    # check in more detail
                    sx, sy = region.contains(super_corners[..., 0],
                                             super_corners[..., 1],
                                             wcs)
                    if len(sx) == 0:
                        # skip this exposure
                        continue
                    wcses.append(wcs)
                    epaths.append(epath)
                    bands.append(band)
                    hdrs.append(self.metastore.headers[band][expID])

        return hdrs, wcses, epaths, bands

    def find_pixels(self, epath, wcs, region):
        """Find all super-pixels in an image described by `hdr` that are within
        a given region, and return lists of the super-pixel data

        Parameters
        -----------
        epath : string
            The path to the exposure data in the HDF5 pixelstore,
            in the form "[band]/[expID]"

        wcs : An instance of astropy.wcs.WCS
            The WCS for the exposure.

        region : An instance of region.Region
            The sky region within which to find pixels

        Returns
        ------------
        data : ndarray of shape (npix,)
            The fluxes of the valid pixels

        ierr : ndarray of dhape (npix,)
            The inverse errors of the valid pixels

        xpix : ndarray of shape (npix,)
            The x pixel coordinates in the exposure of the valid pixels

        ypix : ndarray of shape (npix,)
            The y-pixel coordinates in the exposure of the valid pixels
        """
        s2 = self.pixelstore.super_pixel_size**2
        # this is a (nside, nside, 4, 2) array of the full pixel coordinates of
        # the corners of the superpixels:
        corners = self.pixelstore.superpixel_corners()
        # this returns the superpixel coordinates of every pixel "contained"
        # within a region:
        sx, sy = region.contains(corners[..., 0], corners[..., 1], wcs)
        data = self.pixelstore.data[epath + "/data"][:]
        xpix = self.pixelstore.xpix[sx, sy, :]
        ypix = self.pixelstore.ypix[sx, sy, :]

        return data[sx, sy, :s2], data[sx, sy, s2:], xpix, ypix


class FITSPatch(PixelPatch):

    """This class converts pixel data and meta-data (WCS) information stored in
    FITS files and headers to the data formats required by the device-side code.

    Parameters
    ----------
    fitsfiles : list of string
        Paths to the FITS image and uncertainty data.
    """

    def __init__(self,
                 fitsfiles="",
                 psfstore="",
                 splinedata="",
                 spline_smoothing=None,
                 return_residual=False,
                 sci_ext=0,
                 unc_ext=1,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 debug=0,
                 ):

        super().__init__(return_residual=return_residual,
                         meta_dtype=meta_dtype,
                         pix_dtype=pix_dtype,
                         debug=debug,
                         psfstore=psfstore,
                         splinedata=splinedata,
                         spline_smoothing=spline_smoothing)

        self.fitsfiles = fitsfiles
        self.sci_ext = sci_ext
        self.unc_ext = unc_ext
        self.snr = None
        self.unc = None

    def find_exposures(self, region, bandlist, do_check=True):
        """Return a list of headers (dict-like objects of wcs, filter, and
        exposure id) and exposureIDs for all exposures that overlap the region.
        These should then be sorted by integer band_id.

        Parameters
        ----------
        region : region.Region instance
            Exposures will be found which overlap this region

        bandlist : list of str
            A list of band names to search for images.

        Returns
        -------
        hdrs, wcses, epaths, bands
        """
        fitsfiles = np.array(self.fitsfiles)
        imbands = np.array([fits.getheader(f, self.sci_ext)["FILTER"]
                            for f in fitsfiles])

        epaths, bands, hdrs, wcses = [], [], [], []
        for band in bandlist:
            sel = imbands == band
            if sel.sum() == 0:
                continue
            for i, s in enumerate(sel):
                if not s:
                    continue
                hdr = fits.getheader(fitsfiles[i], self.sci_ext)
                wcs = WCS(hdr)
                # rough check for region coverage
                # NOTE: If bounding box entirely contains image this might fail
                if (region is not None) & do_check:
                    bra, bdec = region.bounding_box
                    imsize = hdr["NAXIS1"], hdr["NAXIS2"]
                    bx, by = wcs.world_to_pixel_values(bra, bdec)
                    inim = np.any((bx > 0) & (bx < imsize[0]) &
                                  (by > 0) & (by < imsize[1]))
                    around_im = ((imsize[0] < bx.max()) & (bx.min() < 0) &
                                 (imsize[1] < by.max()) & (by.min() < 0))
                    inim = inim | around_im
                    if not inim:
                        continue

                epaths.append(fitsfiles[i])
                wcses.append(wcs)
                bands.append(imbands[i])
                hdrs.append(hdr)

        return hdrs, wcses, epaths, bands

    def find_pixels(self, epath, wcs, region):

        # get pixel data, note the transpose
        flux = fits.getdata(epath, self.sci_ext).T
        if getattr(self, "snr", None) is not None:
            ie = self.snr / flux
        elif getattr(self, "unc", None) is not None:
            ie = np.zeros_like(flux) + 1.0 / self.unc
        else:
            ie = 1.0 / fits.getdata(epath, self.unc_ext).T

        bad = ~np.isfinite(flux) | ~np.isfinite(ie) | (ie < 0)
        ie[bad] = 0
        flux[bad] = 0

        nx, ny = flux.shape
        # NOTE: the x,y order swap here is important
        yp, xp = np.meshgrid(np.arange(ny), np.arange(nx))

        # restrict pixels
        if region is None:
            sx, sy = slice(None), slice(None)
        else:
            offsets = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
            lower_left = np.array([xp, yp])
            corners = offsets[:, :, None, None] + lower_left[None, :, :, :]
            corners = corners.transpose(2, 3, 0, 1)
            sx, sy = region.contains(corners[..., 0], corners[..., 1], wcs)

        flux = flux[sx, sy].reshape(-1)
        ie = ie[sx, sy].reshape(-1)
        xp = xp[sx, sy].reshape(-1)
        yp = yp[sx, sy].reshape(-1)

        # Pad to multiples of say 64
        # in fact only every *band* needs to be padded
        # but we are being dumb in this class anyway.
        warp = getattr(self, "warp_size", 64)
        pad = warp - np.mod(len(flux), warp)
        if pad != warp:
            # we give extra pixels ierr=0 (no weight)
            ie = np.append(ie, np.zeros(pad, dtype=ie.dtype))
            flux = np.append(flux, np.zeros(pad, dtype=flux.dtype))
            # we give them x, y = -1, easy to find later.
            xp = np.append(xp, np.zeros(pad, dtype=xp.dtype)-1)
            yp = np.append(yp, np.zeros(pad, dtype=yp.dtype)-1)

        return flux, ie, xp, yp
