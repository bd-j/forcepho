import numpy as np
from forcepho.likelihood import lnlike_multi, make_image
from forcepho.data import PostageStamp
from forcepho import psf as pointspread


class Scene(object):
    """The Scene holds the sources and provides the mapping between a giant 1-d
    array of parameters and the parameters of each source in each band/image
    """

    filterloc = {'F090W': 0}

    def __init__(self, galaxy=False, nfilters=1):

        self.nfilters = nfilters
        if galaxy:
            self.nshape = 4 #ra, dec, q, pa
            self.use_gradients = slice(0, 5)
        else:
            self.nshape = 2 #ra, dec
            self.use_gradients = slice(0, 3)
            
    
    def param_indices(self, sourceid, filterid):
        """Get the indices of the relevant parameters in the giant Theta
        vector, which is assumed to be
        [(fluxes_1), (shape_1), (fluxes_2), (shape_2), ..., (fluxes_n), (shape_n)]
        
        :returns theta:
            An array with elements [flux, (shape_params)]
        """
        start = sourceid * (self.nshape + self.nfilters)
        # get all the shape parameters
        # TODO: nshape (and use_gradients) should probably be an attribute of the source
        inds = range(start + self.nfilters, start + self.nfilters + self.nshape)
        # put in the flux for this source in this band
        inds.insert(0, start + filterid)
        return inds

    def set_source_params(self, theta, source, filterid=None):
        """Set the parameters of a source
        """
        t = np.array(theta).copy()
        if len(t) == 3:
            # Star
            t = np.append(t, np.array([1., 0., 0., 0.]))
        elif len(t) == 5:
            # Galaxy
            t = np.append(np.array(t), np.array([0., 0.]))
        else:
            print("theta vector {} not a valid length: {}".format(theta, len(theta)))
        flux, ra, dec, q, pa, sersic, rh = t
        # if allowing sources to hold the multiband fluxes you'd do this line
        # instead.  Or something even smarter since probably want to update all
        # sources and fluxes at once.
        #source.flux[filterid] = flux
        source.flux = flux
        source.ra = ra
        source.dec = dec
        source.q = q
        source.pa = pa
        source.sersic = sersic
        source.rh = rh

    def set_params(self, Theta, filterid=0):
        """Set all source parameters at once.
        """
        for source in self.sources:
            inds = self.param_indices(source.id, filterid)
            print(inds)
            self.set_source_params(Theta[inds], source, filterid)


#def negative_lnlike_stamp(theta, scene=None, stamp=None):
#    nll, nll_grad = negative_lnlike_multistamp(theta, scene=scene, stamps=[stamp])
#    return nll, nll_grad


def negative_lnlike_nograd(theta, scene=None, stamp=None):
    nll, nll_grad = negative_lnlike_multistamp(theta, scene=scene, stamps=[stamp])
    return nll


def chi_vector(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return chi


def numerical_image_gradients(theta0, delta, scene=None, stamp=None):

    dI_dp = []
    for i, (p, dp) in enumerate(zip(theta0, delta)):
        theta = theta0.copy()
        imlo, _ = make_image(theta, scene, stamp)
        theta[i] += dp
        imhi, _ = make_image(theta, scene, stamp)
        dI_dp.append((imhi - imlo) / (dp))

    return np.array(dI_dp)

            
def make_stamp(size=(100, 100), fwhm=1.0, psfname=None, offset=0.):
    """Make a postage stamp of the given size, including a PSF

    :param size:
        The size in pixels, 2-element tuple

    :param fwhm:
        For a single gaussian PSF, the FWHM of the PSF in pixels

    :param offset:
        The offset of the position of the object from the stamp center.  Useful
        for playing with subpixel offsets.

    :param psfname:
        The path and filename of the gaussian mixture PSF parameters.
    """

    # --- Get a stamp with a give size ----
    stamp = PostageStamp()
    size = np.array(size).astype(int)
    stamp.nx, stamp.ny = size
    stamp.npix = int(stamp.nx * stamp.ny)
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    # override the WCS so coordinates are in pixels
    # The scale matrix D
    stamp.scale = np.eye(2)
    # The sky coordinates of the reference pixel
    stamp.crval = np.zeros([2]) + offset
    # The pixel coordinates of the reference pixel
    stamp.crpix = np.zeros([2])


    # --- Add the PSF ---
    if psfname is not None:
        import pickle
        with open(psfname, 'rb') as pf:
            pdat = pickle.load(pf)

        oversample, pcenter = 8, 504 - 400
        answer = pdat[6][2]
        stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=pcenter)
    else:
        stamp.psf = pointspread.PointSpreadFunction()
        stamp.psf.covariances *= fwhm / 2.355
        
    # --- Add extra information ---
    #stamp.full_header = dict(hdr)
    
    return stamp
