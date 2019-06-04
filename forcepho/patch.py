#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""patch.py -- data model for a patch on the sky
"""

import numpy as np
import h5py

from .gaussmodel import convert_to_gaussians, compute_gaussian
from .sources import Scene

class Patch:

    def __init__(self,
        stamps,            # A list of PostageStamp objects (exposure data) from FITS files
        miniscene,         # All peaks identified in this patch region
        mask,              # The mask that defines the nominal patch
        super_pixel_size=1,  # Number of pixels in each superpixel
        dtype=np.float32   # data type precision for pixel and flux data
        ):
        '''
        Constructs a Patch from PostageStamps (exposures) and a MiniScene
        (a list of pre-identified peaks/sources).  The Patch packs the
        exposures and sources in a manner suitable for sending to the GPU.
        This includes rearranging the data into (padded) super-pixel order.

        The Patch object contains methods for sending the patch data to
        the GPU with PyCUDA.  It also serves as the entry point for sending
        new parameter proposals to the GPU and getting back the results.
        '''

        # Group stamps by band
        # stamp.band must be an int identifier (does not need to be contiguous)
        stamps = sorted(stamps, key=lambda st: st.band)

        bands = [stamp.band for stamp in stamps]
        uniq_bands, n_exp_per_band = np.unique(bands, return_counts=True)

        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = miniscene.n_radii

        # Use the stamps and miniscene to populate these
        self.n_bands = len(uniq_bands)          # Number of bands/filters
        self.n_exp = len(stamps)           # Number of exposures covering the patch (all bands)
        self.n_active = miniscene.n_active         # Number of sources in the patch
        NPHI = 
        NDERIV =          # Number of non-zero Jacobian elements per ImageGaussian
        NTHETA = NBAND + 6, # Number of on-sky parameters per source

        # Pixel Data
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        
        # Pack the 2D stamps into 1D arrays
        pack_pix(stamps, super_pixel_size)

        pack_astrometry(stamps)

        pack_psf(miniscene)
        
        # Here is the on-sky and on-image source information
        # 
        # source_params = np.empty([NSOURCE, NTHETA], dtype=np.float64)
        # source_metadata = np.empty([NSOURCE, n_exp, MANY], dtype=np.float64)

        # Here are the actual on-image and on-sky objects
        # gaussians = np.empty([n_exp, NSOURCE, NGMAX], dtype=object)
        # sources = np.empty([NSOURCE], dtype=object)

        # miniscene sources know their affine transformation values

    def pack_psf(self, miniscene):
        '''
        Each Sersic radius bin has a number of Gaussians associated with it from the PSF.
        The number of these will be constant in a given band, but the Gaussian parameters
        vary with source and exposure.

        We'll just track the total count across radius bins; the individual Gaussians
        will know which bin they belong to.
        '''

        self.n_psf_per_source


    def pack_astrometry(self, sources, dtype=np.float32):
        '''
        The sources know their local astrometric transformation matrices
        (this was previously populated from the stamps).  We need to send
        this data to the GPU so it can apply the sky-to-pixel transformations
        to compare with the image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        - self.G
        '''

        # Each source need different astrometry for each exposure

        self.D = np.empty((self.n_exp, self.n_active, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_active, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, 2), dtype=dtype)
        self.G = np.empty((self.n_exp), dtype=dtype)

        # TODO: are the sources going to have their per-stamp info in the same order that we received the stamps?
        # We already resorted the stamps
        for i in range(self.n_exp):
            # these values are per-exposure
            # TODO: better way to get them than reading the first source?
            self.crpix[i] = sources[0].stamp_crpixs[i]
            self.crval[i] = sources[0].stamp_crvals[i]
            self.G[i] = sources[0].stamp_zps[i]

            for s,source in enumerate(sources):
                self.D[i,s] = source.stamp_scales[i]
                self.CW[i,s] = source.stamp_cds[i] # dpix/dra, dpix/ddec



    def pack_pix(self, stamps, super_pixel_size):
        '''
        We have stamps of exposures that we want to pack into
        concatenated 1D pixel arrays.  We want them to be in
        super-pixel order, too.

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
        '''
        
        # TODO: convert function to numba

        assert super_pixel_size == 1  # TODO: super-pixel ordering

        shapes = np.array([stamp.shape for stamp in stamps])
        sizes = np.array([stamp.size for stamp in stamps])

        total_padded_size = np.sum(sizes)  # will have super-pixel padding

        self.xpix = np.empty(total_padded_size, dtype=dtype)
        self.ypix = np.empty(total_padded_size, dtype=dtype)
        self.data = np.empty(total_padded_size, dtype=dtype)  # value (i.e. flux) in pixel.
        self.ierr = np.empty(total_padded_size, dtype=dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

        i,b = 0,0
        for e,stamp in enumerate(stamps):
            # are we starting a new band?
            if e > 0 and stamp.band != stamp.band[e-1]:
                b += 1
            self.band_N[b] += 1

            N = stamp.size
            self.exposure_start[e] = i
            self.exposure_N[e] = N

            self.xpix[i:i+N] = stamp.xpix
            self.ypix[i:i+N] = stamp.ypix
            self.data[i:i+N] = stamp.vpix
            self.ierr[i:i+N] = stamp.ierr

            # Finished this exposure
            i += N

        assert i == total_padded_size

        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]


    def send_to_gpu(self, residual=False):
        '''
        Transfer the patch data to GPU main memory.  Saves the pointers
        and builds the Patch struct from patch.cu; sends that to GPU memory.
        Saves the pointer for forwarding to the likelihood call.

        Parameters
        ----------
        residual: bool, optional
            Whether to allocate GPU-side space for a residual image array.
            Default: False.
        '''

        self.gpu_patch = 

    def send_proposal(self, proposal, block=1024):
        grid = (n_bands,1,1)
        block = (block,1,1)

        kernel(self.gpu_patch, proposal, results, grid=grid, block=block)




class CPUPatch(Patch):

    def checkin_scene(self):
        """Copy the parameters of the current scene into the Global scene.
        """
        pass


class SquarePatch(Patch):
    
    def get_valid_pixels(self, ex):
        ra_min, ra_max, de_min, de_max = self.sky_region
        sky = pixelcoords_to_skycoords(ex)
        good = ((sky[0] > ra_min) & (sky[0] < ra_max) &
                (sky[1] > de_min) & (sky[1] < de_max))
    
        assert inpatch.sum() > 0
        