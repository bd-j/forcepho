# -*- coding: utf-8 -*-

"""patch.py

Data model for a patch on the sky. The corresponding GPU-side CUDA struct is
in patch.cu.
"""

import numpy as np

from astropy.wcs import WCS

from ..sources import Scene, Galaxy
from ..proposal import Proposer
from .storage import PSF_COLS, PSFStore


__all__ = ["PatchBase", "Patch",
           "scale_at_sky"]

# FIXME: make logic for scene setting and zerocoords more robust.


class PatchBase:

    """Abstract Base class for objects that pack pixel and scene metadata into
    structures in a manner suitable for sending to the GPU. This includes
    rearranging the data into (padded) super-pixel order.

    The Patch object contains methods for sending the patch data to the GPU
    with PyCUDA.

    Parameters
    ----------
    return_residual: bool, optional
        Whether the residual image will be returned from the GPU.
        Default: False.

    pix_dtype: np.dtype, optional
        The Numpy datatype of the pixel data, like fluxes and coordinates.
        Default: np.float32

    meta_dtype: np.dtype, optional
        The Numpy datatype of the non-pixel data, like astrometry.
        Default: np.float32
    """

    # The following must be kept bitwise identical to patch.cu!
    # TODO: compare the dtype size to the struct size
    ptr_dtype = np.uintp  # pointer type
    pdt = np.dtype([('data', ptr_dtype),
                    ('ierr', ptr_dtype),
                    ('xpix', ptr_dtype),
                    ('ypix', ptr_dtype),
                    ('residual', ptr_dtype),

                    ('exposure_start', ptr_dtype),
                    ('exposure_N', ptr_dtype),

                    ('band_start', ptr_dtype),
                    ('band_N', ptr_dtype),

                    ('n_sources', np.int32),
                    ('n_radii', np.int32),

                    ('rad2', ptr_dtype),

                    ('D', ptr_dtype),
                    ('crpix', ptr_dtype),
                    ('crval', ptr_dtype),
                    ('CW', ptr_dtype),

                    ('G', ptr_dtype),

                    ('n_psf_per_source', ptr_dtype),
                    ('psfgauss', ptr_dtype),
                    ('psfgauss_start', ptr_dtype),

                    ], align=True)

    patch_struct_dtype = pdt
    meta_names = ["n_sources", "n_radii", "rad2",
                  "D", "crpix", "crval", "CW", "G",
                  "n_psf_per_source", "psfgauss", "psfgauss_start"]

    def __init__(self,
                 return_residual=False,
                 pix_dtype=np.float32,   # data type for pixel and flux data
                 meta_dtype=np.float32,  # data type for non-pixel data
                 ):
        self.pix_dtype = pix_dtype
        self.meta_dtype = meta_dtype
        self.return_residual = return_residual
        self._dirty_data = True

    @property
    def npix(self):
        try:
            n = self.xpix.shape[0]
            return n
        except(AttributeError):
            return 0

    @property
    def size(self):
        s = 0
        for arrname in self.patch_struct_dtype.names:
            try:
                s += getattr(self, arrname).nbytes
            except(AttributeError):
                pass
        return s

    def clear(self):
        for arrname in self.patch_struct_dtype.names:
            try:
                delattr(self, arrname)
            except(AttributeError):
                pass

    def free(self):
        pass

    def __del__(self):
        self.free()  # do we want to do this?
        self.clear()

    def build_patch(self, npix=1, n_sources=1, n_radii=1,
                    n_bands=1, n_exp=1):
        """This is an abstract method that takes scene and exposure data and
        packs it all into arrays with the proper shapes and data types for
        transfer to GPU.

        This should be overridden by  subclasses, which might get the data in
        different formats.

        Fills in the following arrays:

        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start     [NBAND] exposure index corresponding to the start of each band
        - self.band_N         [NBAND] number of exposures in each band
        - self.exposure_start [NEXP]  pixel index corresponding to the start of each exposure
        - self.exposure_N     [NEXP]  number of pixels (including warp padding) in each exposure

        - self.n_sources     [1]
        - self.n_radii       [1]
        - self.rad2          [N_RADII]

        - self.D             [NEXP, NSOURCE, 2, 2]
        - self.CW            [NEXP, NSOURCE, 2, 2]
        - self.crpix         [NEXP, NSOURCE, 2]
        - self.crval         [NEXP, NSOURCE, 2]
        - self.G             [NEXP]

        - self.n_psf_per_source  [NBAND]  number of PSFGaussians per source in each band
        - self.psfgauss          [NPSFG]  An array of PSFGaussian parameters
        - self.psfgauss_start    [NEXP]   PSFGaussian index corresponding to the start of each exposure.
        """
        self.n_exp = n_exp

        # --- Pixel Data ---
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        self.xpix = np.zeros(npix, dtype=self.pix_dtype)
        self.ypix = np.zeros(npix, dtype=self.pix_dtype)
        self.data = np.zeros(npix, dtype=self.pix_dtype)  # flux
        self.ierr = np.zeros(npix, dtype=self.pix_dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.empty(n_bands, dtype=np.int16)
        self.band_N = np.zeros(n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(n_exp, dtype=np.int32)
        self.exposure_N = np.empty(n_exp, dtype=np.int32)

        # --- Sources ---
        self.n_sources = n_sources
        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = n_radii
        self.rad2 = np.empty(self.n_radii, dtype=self.meta_dtype)

        # --- Meta Data ---
        # Each source need different astrometry for each exposure
        self.D = np.empty((n_exp, self.n_sources, 2, 2), dtype=self.meta_dtype)
        self.CW = np.empty((n_exp, self.n_sources, 2, 2), dtype=self.meta_dtype)
        self.crpix = np.empty((n_exp, self.n_sources, 2), dtype=self.meta_dtype)
        self.crval = np.empty((n_exp, self.n_sources, 2), dtype=self.meta_dtype)
        self.G = np.empty((n_exp), dtype=self.meta_dtype)

        # --- PSF data ---
        # Each source gets a different set of PSF gaussians for each exposure.
        # But, the number of Gaussians per source is constant within a band.
        psf_dtype = np.dtype([('gauss_params', self.meta_dtype, 6),
                              ('sersic_bin', np.int32)])
        self.n_psf_per_source = np.zeros(n_bands, dtype=np.int32)
        n_psfgauss = (self.n_psf_per_source * self.band_N * n_sources).sum()
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        self.psfgauss_start = np.zeros(n_exp, dtype=np.int32)


class Patch(PatchBase):

    """Subclass of PatchBase that includes logic for setting meta data arrays
    given scenes and lists of FITS header dictionaries and WCS instances.
    """

    def __init__(self,
                 psfstore="",
                 splinedata="",
                 return_residual=False,
                 meta_dtype=np.float32,
                 pix_dtype=np.float32,
                 debug=0):

        super().__init__(return_residual=return_residual,
                         meta_dtype=meta_dtype,
                         pix_dtype=pix_dtype)

        self.debug = debug

        self.psfstore = PSFStore(psfstore)
        self.splinedata = splinedata

        self.patch_reference_coordinates = np.zeros(2)
        self.wcs_origin = 0


    def pack_meta(self, scene):
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
        # Set a reference coordinate near center of scene;
        # Subtract this from source coordinates
        self.patch_reference_coordinates = self.zerocoords(scene)
        # Cache number of sources
        self.n_sources = len(scene.sources)

        self._pack_source_metadata(scene)
        self._pack_astrometry(self.wcses, scene)
        self._pack_fluxcal(self.hdrs)
        self._pack_psf(self.bands, self.wcses, scene)
        self.scene = scene  # not really necessary

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
        self.n_sources = scene.n_active

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
        self.crpix = np.empty((self.n_exp, self.n_sources, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, self.n_sources, 2), dtype=dtype)
        # FIXME: this is a little hacky;
        # what if zerocoords hasn't been called after the scene changed?
        ra0, dec0 = self.patch_reference_coordinates

        for j, wcs in enumerate(wcses):
            # TODO - should do a better job getting a matched crpix/crval pair
            # near the center of a patch
            # TODO: Source specific crpix, crval pairs?
            # Using ref coords of patch for crval and zero-indexed crpix
            #self.crval[j] = np.zeros(2, dtype=self.meta_dtype)
            #self.crpix[j] = wcs.all_world2pix(ra0, dec0, 0)
            for i, s in enumerate(scene.sources):
                ssky = np.array([s.ra + ra0, s.dec + dec0])
                CW_mat, D_mat = scale_at_sky(ssky, wcs)
                self.D[j, i] = D_mat
                self.CW[j, i] = CW_mat
                # source specific:
                self.crval[j, i] = ssky - self.patch_reference_coordinates
                self.crpix[j, i] = wcs.all_world2pix(ssky[0], ssky[1], self.wcs_origin)

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

    def set_scene(self, sourcecat, splinedata=None):
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
        if splinedata is None:
            splinedata = self.splinedata
        scene = Scene(catalog=sourcecat, filternames=self.bandlist,
                      source_type=Galaxy, splinedata=splinedata)
        #scene.from_catalog(sourcecat,)
        return scene

    def zerocoords(self, scene, sky_zero=None):
        """Reset (in-place) the celestial zero point of the image metadata and
        the source coordinates to avoid catastrophic cancellation errors in
        coordinates when using single precision.

        Parameters
        ----------
        scene : `sources.Scene()` instance
            A Scene object, where each source has the attributes `ra`, `dec`,

        sky_zero: optional, 2-tuple of float64
            The (ra, dec) values defining the new central coordinates.  These
            will be subtracted from the relevant source and stamp coordinates.
            If not given, the median coordinates of the scene will be used.

        Returns
        -------
        reference_coordinates : ndaray of shape (2,)
            The new reference coordinates, all source positions are offsets
            from this.  This should be subtracted from crval coordinates.
        """
        if sky_zero is None:
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
            source.ra += zero[0]
            source.dec += zero[1]
        #self.patch_reference_coordinates = np.array([0, 0])
        return scene

    def split_pix(self, attr):
        """Split the pixel data into separate arrays for each exposure.

        Returns
        -------
        pixdat : list of ndarray
            List of N_exp ndarrays, each containing the `attr` pixel
            information for that exposure.  List order is the same as
            `self.epaths`
        """
        arr = getattr(self, attr)
        return np.split(arr, np.cumsum(self.exposure_N[:-1]))

    def split_band(self, attr):
        """Split the pixel data into separate arrays for each band.

        Parameters
        ----------
        attr : string
            The Patch attribute to split into separate exposures.  One of
            xpix | ypix | data | ierr

        Returns
        -------
        asplit : list of ndarrays
            List of N_band ndarrays of shape (n_pix_band,)  each containing the
            `attr` pixel information for that band.  List order is the same as
            `self.bandlist`
        """
        arr = getattr(self, attr)
        return np.split(arr, np.cumsum(self.band_N_pix[:-1]))


def scale_at_sky(sky, wcs, dpix=1.0, origin=1, make_approx=False):
    """Get the local linear approximation of the scale and CW matrix at the
    celestial position given by `sky`.  This is a simple numerical calculation

    Parameters
    ---------
    sky : iterable, length 2
        The RA and Dec coordinates in units of degrees at which to compute the
        linear approximation

    wcs : astropy.wcs.WCS() instance
        The wcs to which you want a local linear approximation

    dpix : optional, float, default; 1.0
        The number of pixels to offset to compute the local linear approx

    origin : optiona, default; 1
        The astropy wcs `origin` keyword

    Returns
    --------
    CW_mat : ndarray of shape (2, 2)
        The matrix such that (dx, dy) = CW_mat \dot (dra, ddec) where dx, dy
        are expressed in pixels and dra, ddec are exressed in degrees

    D_mat : ndarray of shape (2, 2)
        The matrix giving pixels per second of arc in RA and DEC.  Equivalent
        to the matrix inverse of 3600 times wcs.pixel_scale_matrix() if there are
        no distortions.
    """
    ra, dec = sky
    # get dsky for step dx, dy = dpix
    if wcs.has_distortion or make_approx:
        pos0_sky = np.array([ra, dec])
        pos0_pix = wcs.all_world2pix([pos0_sky], origin)[0]
        pos1_pix = pos0_pix + np.array([dpix, 0.0])
        pos2_pix = pos0_pix + np.array([0.0, dpix])
        pos1_sky = wcs.all_pix2world([pos1_pix], origin)[0]
        pos2_sky = wcs.all_pix2world([pos2_pix], origin)[0]

        # compute dpix_dsky matrix
        P = np.eye(2) * dpix
        St = np.array([pos1_sky - pos0_sky, pos2_sky - pos0_sky])
        CW_mat = np.linalg.solve(St, P).T

        # compute D matrix
        Winv = np.eye(2)
        Winv[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))**(-1)
        D_mat = 1.0 / 3600.0 * np.matmul(CW_mat, Winv)

    else:
        W = np.eye(2)
        W[0, 0] = np.cos(np.deg2rad(dec))
        D_mat = np.linalg.inv(wcs.pixel_scale_matrix * 3600.0)
        CW_mat = np.matmul(D_mat * 3600.0, W)

    return CW_mat, D_mat
