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
        SuperPixelSize=1,  # Number of pixels in each superpixel
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

        # Use the stamps and miniscene to populate these
        NBAND =           # Number of bands/filters
        NEXP =            # Number of exposures covering the patch
        NSOURCE =          # Number of sources in the patch
        NACTIVE =          # number of active sources in the patch
        NGMAX =           # Maximum number of gaussians per galaxy in any exposure
        NSUPER =          # Number of superpixels
        NPHI =            # Number of on-image parameters per ImageGaussian
        NDERIV =          # Number of non-zero Jacobian elements per ImageGaussian
        NTHETA = NBAND + 6, # Number of on-sky parameters per source

        # Pixel Data
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        pixel_data_shape = [total_padded_size]

        xpix = np.empty(pixel_data_shape, dtype=dtype)
        ypix = np.empty(pixel_data_shape, dtype=dtype)
        vpix = np.empty(pixel_data_shape, dtype=dtype)  # value (i.e. flux) in pixel.
        ierr = np.empty(pixel_data_shape, dtype=dtype)  # 1/sigma

        # Here are arrays that tell you which pixels belong to which bands & exposures.
        exposureIDs = np.empty(NEXP, dtype=np.int16)
        bandIDs = np.empty(NBAND, dtype=np.int16)
        exposure_nsuper = np.empty(NEXP, dtype=np.int64)
        exposure_start = np.cumsum(exposure_nsuper)
        
        # Here is the on-sky and on-image source information
        source_params = np.empty([NSOURCE, NTHETA], dtype=np.float64)
        source_metadata = np.empty([NSOURCE, NEXP, MANY], dtype=np.float64)

        # Here are the actual on-image and on-sky objects
        gaussians = np.empty([NEXP, NSOURCE, NGMAX], dtype=object)
        sources = np.empty([NSOURCE], dtype=object)

    @classmethod
    def make_example():
        example = Patch(
                    NBAND   = 30,
                    NEXP    = 10,
                    NSOURCE = 12,
                    NACTIVE = 12,
                    NGMAX   = 20,
                    NSUPER  = 1, 
                    SuperPixelSize = 1,
                    NPHI    = 6,
                    NDERIV  = 15,
                    )
        return example


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

    def load_pixeldata(self, exposures, SuperPixelSize=1):
        """Populate the pixel data arrays and the pixel ID arrays for a set of
        exposures. This method only keeps pixels that are within the patch.
        """ 
        self.NEXP = len(exposures)
        # for expID, ex in enumerate(exposures):
        #   mask based on region
        #   calculate & store number of superpixels
        #   concatenate (with padding and reshape if necessary)
        self.NSUPER = len(self.xpix)
        self.NBAND = 15 # determine this from the list of filter names

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
        
