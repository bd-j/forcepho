#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""patch.py -- data model for a patch on the sky
"""

import numpy as np
import h5py

from .gaussmodel import convert_to_gaussians, compute_gaussian
from .sources import Scene

class Patch(object):
    

    # Sizes of things
    NBAND   = 1         # Number of bands/filters
    NEXP    = 1         # Number of exposures covering the patch
    NSOURCE = 1         # Number of sources in the patch
    NACTIVE = 1         # number of active sources in the patch
    NGMAX   = 20        # Maximum number of gaussians per galaxy in any exposure
    NSUPER  = 1         # Number of superpixels total (all exposures)
    SuperPixelSize = 1  # Number of pixels in each superpixel
    NPHI    = 6         # Number of on-image parameters per ImageGaussian
    NTHETA  = NBAND + 6 # Number of on-sky parameters per source
    NDERIV  = 15        # Number of non-zero Jacobian elements per ImageGaussian

    # Pixel Data
    # These are arrays of pixel data for *all* pixels in a patch (i.e.
    # multiple exposures)
    pixel_data_shape = [NSUPER, SuperPixelSize]

    xpix = np.empty(pixel_data_shape, dtype=np.float32)
    ypix = np.empty(pixel_data_shape, dtype=np.float32)
    vpix = np.empty(pixel_data_shape, dtype=np.float32)  # value (i.e. flux) in pixel.
    ierr = np.empty(pixel_data_shape, dtype=np.float32)  # 1/sigma

    # Here are arrays that tell you which pixels belong to which exposures.
    exposureIDs = np.empty(NEXP, dtype=np.int16)
    exposure_nsuper = np.empty(NEXP, dtype=np.int64)
    exposure_start = np.cumsum(exposure_nsuper)
    
    # Here is the on-sky and on-image source information
    gaussian_params = np.empty([NEXP, NSOURCE, NGMAX, NPHI], dtype=np.float64)
    gaussian_jacobians = np.empty([NEXP, NSOURCE, NGMAX, NDERIV], dtype=np.float64)
    source_params = np.empty([NSOURCE, NTHETA], dtype=np.float64)
    source_metadata = np.empty([NSOURCE, NEXP, MANY], dtype=np.float64)

    # Here are the actual on-image and on-sky objects
    gaussians = np.empty([NEXP, NSOURCE, NGMAX], dtype=object)
    sources = np.empty([NSOURCE], dtype=object)


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
        #   calculate & store lengths
        #   concatenate (with padding and reshape if necessary)
        self.NSUPER = len(self.xpix)
        self.NBAND = 15 # determine this from the list of filter names

    def load_metadata(self, exposures):
        """Add metadata for each exposure to each source.
        """
        for expID, ex in enumerate(exposures):
            for s in self.scene.sources:
                # Do this for CRVAL, CRPIX, dpix_dsky, scale, and psf info.
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


class GPUPatch(Patch):

    def convert_to_gaussians(self, blockID):
        expID = self.exposureIDs[blockID]
        for sourceID in range(self.NSOURCES):
            activeID = self.ActiveSourceIDs[sourceID]
            gig = convert_to_gaussians(self.sources[sourceID], expID, 
                                        compute_deriv=(activeID >= 0))
            self.gaussians[expID, sourceID, :] = gig.gaussians

    def process_pixel(self, blockID=slice(None), threadID=0):

        xpix = self.xpix[blockID, threadID]
        ypix = self.ypix[blockID, threadID]
        isig = self.ierr[blockID, threadID]
        delt = self.vpix[blockID, threadID]
        
        # get the exposure ID for the pixels in this block
        ind = np.searchsorted(self.exposure_start, blockID)
        expID = self.exposureIDs[ind]
    
        # Get the residual
        for sourceID in range(self.NSOURCES):
            for gaussID in range(self.NGMAX):
                g = self.gaussians[expID, sourceID, gaussID]
                if g is None:
                    continue
                delt -= compute_gaussian(g, xpix, ypix, compute_deriv=False)
        
        # compute the chi value and accumulate chi^2
        chi = delt * ierr
        self.CHISQ += chi * chi

        # Get the gradients    
        for sourceID in range(self.nsources):
            # which active source (with varied parameters) is this?
            activeID = self.ActiveSourcesID[sourceID]
            if activeID < 0:
                # skip inactive sources
                continue
            for gaussID in range(self.ngmax):
                g = gaussians[expID, sourceID, gaussID]
                if g is None:
                    continue
                # get the vector of 6 image gradients w/r/t image params
                _, dI_dphi = compute_gaussian(g, xpix, ypix, compute_deriv=True)
                # convert to gradients w/r/t scene params
                dI_dtheta = apply_jacobian(g, dI_dphi)
                # accumulate
                self.DCHISQ_DSCENE[activeID, g.bandID] += chi * ierr *dI_dtheta[0]
                self.DCHISQ_DSCENE[activeID, self.NBANDS:] += chi * ierr *dI_dtheta[1:]

    def lnlike(self):
        self.CHISQ = 0
        self.DCHISQ_DSCENE = np.zeros([self.NACTIVE, self.NBAND + 6])
        self.convert_to_gaussians()
        self.process_pixel()
        
        return self.CHISQ, self.DCHISQ_DSCENE.reshape(-1)


def apply_jacobian(g, dI_dphi):
    """Apply the jacobian for a given image gaussian to convert from gradients
     with respect to on-image parameters to gradients with respect to on-scene
     parameters.  Generically this is a matrix multiply, but there are many
     zeros in the Jacobian that we don't necessarily want to carry around so the
     matrix maultiply might be done more explicitly.
    """
    return np.matmul(g.derivs, dI_dphi)






class SquarePatch(Patch):
    
    def get_valid_pixels(self, ex):
        ra_min, ra_max, de_min, de_max = self.sky_region
        sky = pixelcoords_to_skycoords(ex)
        good = ((sky[0] > ra_min) & (sky[0] < ra_max) &
                (sky[1] > de_min) & (sky[1] < de_max))
    
        assert inpatch.sum() > 0
        
