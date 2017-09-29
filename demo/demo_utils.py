import numpy as np
from forcepho.likelihood import model_image
from forcepho.data import PostageStamp
from forcepho import psf as pointspread


class Scene(object):

    def set_params(self, theta):
        # Add all the unused (fixed) galaxy parameters
        t = np.array(theta).copy()
        if len(t) == 3:
            # Star
            self.params = [np.append(t, np.array([1., 0., 0., 0.]))]
            self.free_inds = slice(0, 3)
        elif len(t) == 5:
            # Galaxy
            self.params = [np.append(np.array(t), np.array([0., 0.]))]
            self.free_inds = slice(0, 5)
        else:
            print("theta vector {} not a valid length: {}".format(theta, len(theta)))


def negative_lnlike_stamp(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials[scene.free_inds, :], axis=-1)


def negative_lnlike_nograd(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2)


def chi_vector(theta, scene=None, stamp=None):
    stamp.residual = stamp.pixel_values.flatten()
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return chi


def make_image(theta, scene=None, stamp=None):
    stamp.residual = np.zeros(stamp.npix)
    scene.set_params(theta)
    sources, thetas = scene.sources, scene.params
    residual, partials = model_image(thetas, sources, stamp)
    return -residual.reshape(stamp.nx, stamp.ny), partials


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
    # The distortion matrix D
    stamp.distortion = np.eye(2)
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
