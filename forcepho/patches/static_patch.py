# -*- coding: utf-8 -*-

"""patch.py

Data model for a patch on the sky. The corresponding GPU-side CUDA struct is
in patch.cu.
"""

import numpy as np
from .patch import Patch


__all__ = ["StaticPatch", "patch_to_stamps"]


class StaticPatch(Patch):

    """A patch where the data is built from area matched PostageStamp and
    Scene objects for which most of the required metadata has already been
    tabulated as attributes of these objects.  These are typically the output
    of `patch_conversion.py`
    """


    def __init__(self,
        stamps,            # A list of PostageStamp objects (exposure data) from FITS files
        miniscene,         # All peaks identified in this patch region
        mask=None,              # The mask that defines the nominal patch
        super_pixel_size=1,  # Number of pixels in each superpixel
        return_residual=False,
        pix_dtype=np.float32,  # data type for pixel and flux data
        meta_dtype=np.float32,   # data type for non-pixel data
        ):
        """
        Constructs a Patch from PostageStamps (exposures) and a MiniScene
        (a list of pre-identified peaks/sources).  The Patch packs the
        exposures and sources in a manner suitable for sending to the GPU.
        This includes rearranging the data into (padded) super-pixel order.
        The Patch object contains methods for sending the patch data to
        the GPU with PyCUDA.
        Parameters
        ----------
        return_residual: bool, optional
            Whether the residual image will be returned from the GPU.
            Default: False.

        pix_dtype: np.dtype, optional
            The Numpy datatype of the pixel data, like fluxes and coordinates.
            Default: np.float32

        meta_dtype: np.dtype, optional
            The Numpy datatype of the non-pixel data, like transformation matrices.
            Default: np.float32
        """

        self.pix_dtype = pix_dtype
        self.meta_dtype = meta_dtype

        self.return_residual = return_residual

        # Group stamps by band
        # stamp.band must be an int identifier (does not need to be contiguous)
        band_ids = [st.band for st in stamps]
        assert (np.diff(band_ids) >= 0).all(), 'Stamps must be sorted by band'

        bands = [stamp.band for stamp in stamps]
        uniq_bands, n_exp_per_band = np.unique(bands, return_counts=True)

        # Use the stamps and miniscene to populate these
        self.n_bands = len(uniq_bands)          # Number of bands/filters
        self.n_exp = len(stamps)           # Number of exposures covering the patch (all bands)

        # Pixel Data
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)

        # Pack the 2D stamps into 1D arrays
        self.pack_pix(stamps, mask)

        self.pack_source_metadata(miniscene)

        self.pack_astrometry(miniscene.sources)

        self.pack_psf(miniscene)

    def build_patch(self,
                    stamps=[],
                    miniscene=None,  # All peaks identified in this patch region
                    ):
        """Constructs a Patch from PostageStamps (exposures) and a MiniScene
        (a list of pre-identified peaks/sources).  The Patch packs the
        exposures and sources in a manner suitable for sending to the GPU.
        This includes rearranging the data into (padded) super-pixel order.

        The Patch object contains methods for sending the patch data to
        the GPU with PyCUDA.

        Parameters
        -----------

        stamps : list of PostageStamp objects.
            All pixels from these stamps (padded to match GPU warp size) will
            be sent to GPU. They should therefore cover similar regions of the
            sky.

        miniscene : Scene object.
            This is effectively a list of Source objects for active sources in
            the area of the sky covered by the stamps.  It must have the
            following attributes in addition to a normal Scene object:
                * `npsf_per_source`          [N_BAND]
                * `sources[s].stamp_psfs`    [N_EXP]
                * `sources[0].radii`         [N_RADII]
                * `sources[0].stamp_crpixs`  [N_EXP]
                * `sources[0].stamp_crvals`  [N_EXP]
                * `sources[0].stamp_scales`  [N_EXP]
                * `sources[s].stamp_cds`     [N_EXP]
                * `sources[s].stamp_zps`     [N_EXP]
        """
        # --- Group stamps by band ---
        # stamp.band must be an int identifier (does not need to be contiguous)
        band_ids = [st.band for st in stamps]
        assert (np.diff(band_ids) >= 0).all(), 'Stamps must be sorted by band'
        bands = [stamp.band for stamp in stamps]
        uniq_bands, n_exp_per_band = np.unique(bands, return_counts=True)

        # Use the stamps and miniscene to populate these
        self.n_bands = len(uniq_bands)  # Number of bands/filters
        self.n_exp = len(stamps)        # Number of exposures covering the patch (all bands)

        # Pack the 2D stamps into 1D arrays
        self.pack_pix(stamps)

        self.pack_source_metadata(miniscene)
        self.pack_astrometry(miniscene.sources)
        self.pack_psf(miniscene)

    def pack_psf(self, miniscene, dtype=None):
        """Each Sersic radius bin has a number of Gaussians associated with it
        from the PSF. The number of these will be constant in a given band, but
        the Gaussian parameters vary with source and exposure.

        We'll just track the total count across radius bins; the individual
        Gaussians will know which bin they belong to.

        Fills in the following arrays:
        - self.n_psf_per_source  [NBAND]  number of PSFGaussians per source in each band
        - self.psfgauss          [NPSFG]  An array of PSFGaussian parameters
        - self.psfgauss_start    [NEXP]   PSFGaussian index corresponding to the start of each exposure.
        """
        if not dtype:
            dtype = self.meta_dtype

        self.n_psf_per_source = np.empty(self.n_bands, dtype=np.int32)
        self.n_psf_per_source[:] = miniscene.npsf_per_source  # ???

        psf_dtype = np.dtype([('gauss_params', dtype, 6),
                              ('sersic_bin', np.int32)])
        n_psfgauss = self.n_psf_per_source.sum() * self.n_exp * self.n_sources
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)

        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)
        _n_psf_per_source = np.repeat(self.n_psf_per_source, self.band_N)
        self.psfgauss_start[1:] = np.cumsum(_n_psf_per_source)[:-1]*self.n_sources

        s = 0
        for e in range(self.n_exp):
            for source in miniscene.sources:
                # sources have one set of psf gaussians per exposure
                # length of that set is const in a band, however
                this_psfgauss = source.psfgauss(e)
                N = len(this_psfgauss)
                #print(N)
                self.psfgauss[s: s+N] = this_psfgauss
                s += N
            if e < (self.n_exp - 1):
                # check we got our indexing right
                assert s == self.psfgauss_start[e+1], (e,s,self.n_sources,self.psfgauss_start[e+1])

    def pack_source_metadata(self, miniscene, dtype=None):
        """We don't actually pack sources in the Patch; that happens
        in a Proposal.  But we do have some global constants related to
        sources, such as the total number of soures and number of Sersic
        radii bins.  So we pack those here.

        Fills in:
        - self.n_sources
        - self.n_radii
        - self.rad2
        """
        if not dtype:
            dtype = self.meta_dtype

        # number of sources in the patch
        self.n_sources = miniscene.nactive

        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = len(miniscene.sources[0].radii)

        self.rad2 = np.empty(self.n_radii, dtype=dtype)
        self.rad2[:] = miniscene.sources[0].radii**2

    def pack_astrometry(self, sources, dtype=None):
        """The sources know their local astrometric transformation matrices
        (this was previously populated from the stamps).  We need to send this
        data to the GPU so it can apply the sky-to-pixel transformations to
        compare with the image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        - self.G
        """
        if not dtype:
            dtype = self.meta_dtype

        # Each source need different astrometry for each exposure
        self.D = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, self.n_sources, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, self.n_sources, 2), dtype=dtype)
        self.G = np.empty((self.n_exp), dtype=dtype)

        # The ordering of the astrometric information in the sources is
        # guaranteed to be in the same order as our exposures

        for i in range(self.n_exp):
            # we give the same value to every source in the exp (no distortions)
            self.crpix[i, :, :] = sources[0].stamp_crpixs[i]
            self.crval[i, :, :] = sources[0].stamp_crvals[i]
            self.G[i] = sources[0].stamp_zps[i]

            for s, source in enumerate(sources):
                self.D[i, s] = source.stamp_scales[i]
                self.CW[i, s] = source.stamp_cds[i]  # dpix/dra, dpix/ddec

    def pack_pix(self, stamps, dtype=None):
        """We have stamps of exposures that we want to pack into
        concatenated 1D pixel arrays.  We want them to be in super-pixel order,
        too.

        As we go, we want to build up the index arrays that
        allow us to find an exposure in the 1D arrays.

        Fills the following arrays:
        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start
        - self.band_N
        - self.exposure_start
        - self.exposure_N
        """
        if not dtype:
            dtype = self.pix_dtype

        shapes = np.array([stamp.shape for stamp in stamps], dtype=int)
        sizes = np.array([stamp.npix for stamp in stamps], dtype=int)

        self.original_shapes = shapes
        self.original_sizes = sizes

        # will have super-pixel padding
        warp_size = 32
        total_padded_size = np.sum( (sizes + warp_size - 1)//warp_size*warp_size )

        # TODO: we use zeros instead of empty for the padding bytes
        self.xpix = np.zeros(total_padded_size, dtype=dtype)
        self.ypix = np.zeros(total_padded_size, dtype=dtype)
        self.data = np.zeros(total_padded_size, dtype=dtype)  # flux
        self.ierr = np.zeros(total_padded_size, dtype=dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

        i, b = 0, 0
        for e, stamp in enumerate(stamps):
            # are we starting a new band?
            if e > 0 and stamp.band != stamps[e-1].band:
                b += 1
            self.band_N[b] += 1

            N = stamp.npix
            self.exposure_start[e] = i
            warp_padding = (warp_size - N%warp_size)%warp_size
            self.exposure_N[e] = N + warp_padding

            self.xpix[i: i+N] = stamp.xpix.flat
            self.ypix[i: i+N] = stamp.ypix.flat
            self.data[i: i+N] = stamp.pixel_values.flat
            self.ierr[i: i+N] = stamp.ierr.flat

            # Finished this exposure
            i += N + warp_padding

        assert i == total_padded_size

        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]


def patch_to_stamps(patch):
    pixdat = ["xpix", "ypix", "data", "ierr"]
    stamps = [PostageStamp() for i in range(patch.band_N.sum())]
    splits = [np.split(getattr(patch, arr), np.cumsum(patch.exposure_N)[:-1])
              for arr in pixdat]

    for e, stamp in enumerate(stamps):
        stamp.xpix = splits[0][e]
        stamp.ypix = splits[1][e]
        stamp.pixel_values = splits[2][e]
        stamp.ierr = splits[3][e]

        # Add metadata from first source
        stamp.crpix = patch.crpix[e, 0]
        stamp.crpix = patch.crval[e, 0]

    scene = patch.scene
    for s, source in enumerate(scene.sources):
        source.stamp_scales = patch.D[:, s, :, :]
        source.stamp_cds = patch.CW[:, s, :, :]  # dpix/dra, dpix/ddec
        source.stamp_crpixs = patch.crpix[:, 0, :]
        source.stamp_crvals = patch.crval[:, 0, :]
        source.stamp_zps = patch.G[:]

        # need to do something about PSFs and about Band ids
        #source.stamp_psfs = 

    return stamps, scene