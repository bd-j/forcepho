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
        thread_block_size=1024,  # Number of threads per GPU thread block
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
        stamps = sorted(stamps, key=lambda st: st.band)

        bands = [stamp.band for stamp in stamps]
        uniq_bands, nexp_per_band = np.unique(bands, return_counts=True)

        max_psf_ngauss = max(stamp.psf.ngauss for stamp in stamps)
        max_gal_ngauss = max(source.ngauss for source in miniscene.sources)

        # Use the stamps and miniscene to populate these
        NBAND = len(uniq_bands)          # Number of bands/filters
        NEXP = len(stamps)           # Number of exposures covering the patch
        NSOURCE = len(miniscene)         # Number of sources in the patch
        NACTIVE = miniscene.nactive         # number of active sources in the patch
        NPHI = 
        NDERIV =          # Number of non-zero Jacobian elements per ImageGaussian
        NTHETA = NBAND + 6, # Number of on-sky parameters per source

        # Pixel Data
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        
        # Pack the 2D stamps into 1D arrays
        pack_pix(stamps, super_pixel_size, thread_block_size)
        
        # Here is the on-sky and on-image source information
        source_params = np.empty([NSOURCE, NTHETA], dtype=np.float64)
        source_metadata = np.empty([NSOURCE, NEXP, MANY], dtype=np.float64)

        # Here are the actual on-image and on-sky objects
        gaussians = np.empty([NEXP, NSOURCE, NGMAX], dtype=object)
        sources = np.empty([NSOURCE], dtype=object)


    def pack_pix(self, stamps, super_pixel_size, thread_block_size):
        '''
        We have stamps of exposures that we want to pack into
        concatenated 1D pixel arrays.  We want them to be in
        super-pixel order, too.

        As we go, we want to build up the index arrays that
        allow us to find an exposure in the 1D arrays.
        '''
        
        # TODO: convert function to numba

        assert super_pixel_size == 1  # TODO: super-pixel ordering

        shapes = np.array([stamp.shape for stamp in stamps])
        sizes = np.array([stamp.size for stamp in stamps])

        total_padded_size = (sizes + thread_block_size - 1)//thread_block_size*thread_block_size

        self.xpix = np.empty(total_padded_size, dtype=dtype)
        self.ypix = np.empty(total_padded_size, dtype=dtype)
        self.vpix = np.empty(total_padded_size, dtype=dtype)  # value (i.e. flux) in pixel.
        self.ierr = np.empty(total_padded_size, dtype=dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.zeros(self.NBAND, dtype=np.int16)
        self.band_N = np.zeros(self.NBAND, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.zeros(self.NEXP, dtype=np.int32)
        self.exposure_N = np.zeros(self.NEXP, dtype=np.int32)

        i,b = 0,0
        for e,stamp in enumerate(stamps):
            self.band_N[b] += 1
            if(e > 0 and stamp.band != stamps[e-1].band)
                b += 1

            N = stamp.size
            self.exposure_N[e] = N

            self.xpix[i:i+N] = stamp.xpix
            self.ypix[i:i+N] = stamp.ypix
            self.vpix[i:i+N] = stamp.vpix
            self.ierr[i:i+N] = stamp.ierr

            # Finished this exposure; pad the pixel array to the thread_block_size
            i += N + N%thread_block_size

        # The start arrays are for convenience
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]
        self.exposure_start[1:] = np.cumsum(self.exposure_N)[:-1]

    def send_to_gpu():
        '''
        Transfer all the pixel data to GPU main memory.
        '''


class CPUPatch(Patch):

    def __init__(self, region, NGMAX=20):
        self.sky_region = region
        self.checkout_scene()
        self.NGMAX = NGMAX

    def checkout_scene(self, sources=None):
        """Find all the sources in this patch in the global Scene, and assign 
        them to be active or inactive, then make a subscene for this patch.
        """
        self.sources = sources
        self.NSOURCE = len(self.sources)
        self.ActiveSourceIDs = np.ones(self.NSOURCE) - 1
        self.scene = Scene(sources)

    def checkin_scene(self):
        """Copy the parameters of the current scene into the Global scene.
        """
        pass

    def load_metadata(self, exposures):
        """Add metadata for each exposure to each source.
        """
        for expID, ex in enumerate(exposures):
            for s in self.scene.sources:
                # Do this for ZP, CRVAL, CRPIX, dpix_dsky, scale, and psf info.
                s.metadata[expID] = ex.metadata

    def get_valid_pixels(self, exposure):
        raise NotImplementedError

    def send_pixel_data(self):
        """Send the pixel data over to the GPU.  This is done only once at the
        beginning.
        """
        # send pixel data
        pass

    def send_sources(self, Theta=None):
        """Send over the sources (with appropriate parameters).  This is done 
        every likelihood call.  Actually we will probably want to send the
        sources once and then just send arrays of parameters.
        """
        if Theta is not None:
            self.scene.set_all_parameters(Theta)
        # send self.scene.sources


class SquarePatch(Patch):
    
    def get_valid_pixels(self, ex):
        ra_min, ra_max, de_min, de_max = self.sky_region
        sky = pixelcoords_to_skycoords(ex)
        good = ((sky[0] > ra_min) & (sky[0] < ra_max) &
                (sky[1] > de_min) & (sky[1] < de_max))
    
        assert inpatch.sum() > 0
        
