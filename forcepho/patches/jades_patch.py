#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.wcs import WCS

from ..sources import Scene, Galaxy
from ..stamp import scale_at_sky

from .storage import MetaStore, PixelStore, PSFStore
from .storage import PSF_COLS
from .patch import Patch

JWST_BANDS = ["F090W", "F115W", "F150W", "F200W",
              "F277W", "F335M", "F356W", "F410M", "F444W"]

# FIXME: make logic for scene setting and zerocoords more robust.


class JadesPatch(Patch):

    """This class converts between JADES-like exposure level pixel data,
    meta-data (WCS), and PSF information to the data formats required by the
    GPU-side code.

    Parameters
    ----------

    Important Attributes
    ----------

    bandlist : list of str

    scene : forcepho.Scene()
    """

    def __init__(self,
                 pixelstore="",
                 metastore="",
                 psfstore="",
                 splinedata="",
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 ):

        self.meta_dtype = meta_dtype
        self.pix_dtype = pix_dtype
        self.return_residual = return_residual

        self.metastore = MetaStore(metastore)
        self.psfstore = PSFStore(psfstore)
        self.pixelstore = PixelStore(pixelstore)
        self.splinedata = splinedata

        self.patch_reference_coordinates = np.zeros(2)
        self.wcs_origin = 0

    def build_patch(self, region, sourcecat, allbands=JWST_BANDS):
        """Given a ragion and a source catalog, this method finds and packs up
        all the relevant meta- and pixel-data in a format suitable for transfer
        to the GPU

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
            corresponding to keys of the pixel and meta stores.
        """
        # --- Find relevant exposures ---
        # The output lists are all of length n_exp and should all be in band
        # order
        meta = self.find_exposures(region, allbands)
        self.hdrs, self.wcses, self.epaths, self.bands = meta

        # --- Get BAND information for the exposures ---
        # band_ids must be an int identifier (does not need to be contiguous)
        band_ids = [allbands.index(b) for b in self.bands]
        assert (np.diff(band_ids) >= 0).all(), 'Exposures must be sorted by band'
        u, n = np.unique(band_ids, return_counts=True)
        self.uniq_bands, self.n_exp_per_band = u, n
        self.bandlist = [allbands[i] for i in self.uniq_bands]

        # --- Cache some useful numbers ---
        self.n_bands = len(self.uniq_bands)       # Number of bands/filters
        self.n_exp = len(self.hdrs)               # Number of exposures

        # --- Pack up all the data for the gpu ---
        if sourcecat is not None:
            self.pack_meta(sourcecat)
        self.pack_pix(region)

    def pack_meta(self, sourcecat):
        """This method packs all the exposure and source metadata.  Most of
        this data is scene/source dependent, so in this way we can change the
        scene without repacking the pixel data.  This requires the following
        attributes to have been set:
        * uniq_bands      [NBAND]
        * bandlist        [NBAND]
        * n_exp_per_band  [ NBAND]
        * n_exp           [1]
        * wcses           [NEXP]
        * hdrs            [NEXP]
        * bands           [NEXP]

        Parameters
        ----------
        sourcecat : ndarray of shape (n_sources,)
            A structured array of source parameters.  Field names must correspond
            to the forcepho native parameter names, with fluxes for each band in
            their own column.
        """
        # --- Set the scene ---
        # build scene from catalog
        self.scene = self.set_scene(sourcecat)
        # Set a reference coordinate near center of scene;
        # Subtract this from source coordinates
        self.patch_reference_coordinates = self.zerocoords(self.scene)
        # Cache number of sources
        self.n_sources = len(self.scene.sources)

        self._pack_source_metadata(self.scene)
        self._pack_astrometry(self.wcses, self.scene)
        self._pack_fluxcal(self.hdrs)
        self._pack_psf(self.bands, self.wcses, self.scene)

    def _pack_source_metadata(self, scene, dtype=None):
        """We don't actually pack sources in the Patch; that happens
        in a Proposal.  But we do have some global constants related to
        sources, such as the total number of soures and number of Sersic
        radii bins.  So we pack those here.

        Fills in:
        - self.n_sources
        - self.n_radii
        - self.rad2

        Parameters
        ---------
        scene : A forcepho.sources.Scene() instance
        """

        if not dtype:
            dtype = self.meta_dtype

        # number of sources in the patch
        self.n_sources = scene.nactive

        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = len(scene.sources[0].radii)

        self.rad2 = np.empty(self.n_radii, dtype=dtype)
        self.rad2[:] = scene.sources[0].radii**2

    def _pack_astrometry(self, wcses, scene, dtype=None):
        """The sources need to know their local astrometric transformation
        matrices (and photometric conversions) in each exposure. We need to
        calculate these from header/meta information and send data to the GPU
        so it can apply the sky-to-pixel transformations to compare with the
        image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        """
        if not dtype:
            dtype = self.meta_dtype

        self.D = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, 2), dtype=dtype)
        # FIXME: this is a little hacky;
        # what if zerocoords hasn't been called after the scene changed?
        ra0, dec0 = self.patch_reference_coordinates

        for j, wcs in enumerate(wcses):
            # TODO - should do a better job getting a matched crpix/crval pair
            # near the center of a patch
            # TODO: Source specific crpix, crval pairs?
            # Using ref coords of patch for crval and zero-indexed crpix
            self.crval[j] = np.zeros(2, dtype=self.meta_dtype)
            self.crpix[j] = wcs.all_world2pix(ra0, dec0, 0)
            for i, s in enumerate(scene.sources):
                ssky = np.array([s.ra + ra0, s.dec + dec0])
                CW_mat, D_mat = scale_at_sky(ssky, wcs)
                self.D[j, i] = D_mat
                self.CW[j, i] = CW_mat
                # source specific:
                #self.crval[j, i] = ssky - self.patch_reference_coordinates
                #self.crpix[j, i] = wcs.all_world2pix(ssky[0], ssky[1], origin=0)

    def _pack_fluxcal(self, hdrs, tweakphot=None, dtype=None):
        """A nominal flux calibrartion has been applied to all images,
        but here we allow for tweaks to the flux calibration.

        Fills in the following array:
        - self.G
        """
        if not dtype:
            dtype = self.meta_dtype

        self.G = np.ones((self.n_exp), dtype=dtype)
        if not tweakphot:
            return
        else:
            for j, hdr in enumerate(hdrs):
                self.G[j] = hdr[tweakphot]

    def _pack_psf(self, bands, wcses, scene, dtype=None, psf_dtype=None):
        """Each Sersic radius bin has a number of Gaussians associated with it
        from the PSF. The number of these will be constant in a given band, but
        the Gaussian parameters vary with source and exposure.

        We'll just track the total count across radius bins; the individual
        Gaussians will know which bin they belong to.

        Fills in the following arrays:
        - self.n_psf_per_source   [NBAND]  number of PSFGaussians per source in each band
        - self.psfgauss           [NPSFG]  An array of PSFGaussian parameters
        - self.psfgauss_start     [NEXP]   PSFGaussian index corresponding to the start of each exposure.
        """
        if not dtype:
            dtype = self.meta_dtype
        if not psf_dtype:
            psf_dtype = np.dtype([(c, dtype) for c in PSF_COLS] +
                                 [("sersic_bin", np.int32)])

        # Get number of gaussians per source for each band
        npsf_per_source = [self.psfstore.data[b].attrs["n_psf_per_source"]
                           for b in self.bandlist]
        self.n_psf_per_source = np.empty(self.n_bands, dtype=np.int32)
        self.n_psf_per_source[:] = np.array(npsf_per_source)

        # Make array for PSF parameters and index into that array
        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)
        n_psfgauss = (self.n_psf_per_source * self.n_exp_per_band * self.n_sources).sum()
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        s = 0
        for e, wcs in enumerate(wcses):
            self.psfgauss_start[e] = s
            for i, source in enumerate(scene.sources):
                # sources have one set of psf gaussians per exposure
                # length of that set is const in a band, however
                psfparams = self.psfstore.get_local_psf(band=bands[e], source=source)
                N = len(psfparams)
                self.psfgauss[s: (s + N)] = psfparams
                s += N
        assert s == n_psfgauss

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

        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

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
            assert n_pix > 0, "There were no valid pixels in exposure {}".format(self.epaths[e])
            if e > 0 and self.bands[e] != self.bands[e - 1]:
                b += 1
            self.band_N[b] += 1
            self.exposure_start[e] = i
            self.exposure_N[e] = n_pix
            data.append(pixdat[0])
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
                bx, by = wcs.all_world2pix(bra, bdec, self.wcs_origin)
                inim = np.any((bx > 0) & (bx < imsize[0]) &
                              (by > 0) & (by < imsize[1]))
                if inim:
                    # check in more detail
                    sx, sy = region.contains(super_corners[..., 0],
                                             super_corners[..., 1], wcs,
                                             origin=self.wcs_origin)
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
        data : ndrarry of shape (npix,) 
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
        sx, sy = region.contains(corners[..., 0], corners[..., 1],
                                 wcs, origin=self.wcs_origin)
        data = self.pixelstore.data[epath + "/data"][:]
        xpix = self.pixelstore.xpix[sx, sy, :]
        ypix = self.pixelstore.ypix[sx, sy, :]

        return data[sx, sy, :s2], data[sx, sy, s2:], xpix, ypix

    def set_scene(self, sourcecat):
        """Build a scene made of sources with the appropriate filters using

        Parameters
        ----------
        sourcecat : structured ndarray of shape (n_sources,)
            A structured array of source parameters.  The fields of the of the
            structured array must correspond to the fundamental parameters of
            the `Galaxy` sourcetype ('ra', 'dec', 'q', 'pa', 'sersic', 'rhalf')
            as well as have flux fields with the same column names as
            `self.bandlist`.

        Returns
        -------
        scene : An instance of sources.Scene
        """
        scene = Scene()
        scene.from_catalog(sourcecat, filternames=self.bandlist,
                           source_type=Galaxy, splinedata=self.splinedata)
        return scene

    def zerocoords(self, scene, sky_zero=None):
        """Reset (in-place) the celestial zero point of the image metadata and
        the source coordinates to avoid catastrophic cancellation errors in
        coordinates when using single precision.

        Parameters
        ----------
        scene:
            A Scene object, where each source has the attributes `ra`, `dec`,

        sky_zero: optional, 2-tuple of float64
            The (ra, dec) values defining the new central coordinates.  These
            will be subtracted from the relevant source and stamp coordinates.
            If not given, the median coordinates of the scene will be used.
        """
        if not sky_zero:
            zra = np.median([s.ra for s in scene.sources])
            zdec = np.median([s.dec for s in scene.sources])
            sky_zero = np.array([zra, zdec])

        zero = np.array(sky_zero)
        self.patch_reference_coordinates = zero
        for source in scene.sources:
            source.ra -= zero[0]
            source.dec -= zero[1]

        return zero
        # now subtract from all pixel metadata
        #self.crval -= zero[None, :]

    def unzerocoords(self, scene):
        zero = self.patch_reference_coordinates
        for source in scene.sources:
            source.ra -= zero[0]
            source.dec -= zero[1]
