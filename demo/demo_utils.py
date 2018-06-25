import numpy as np
import time

import astropy.io.fits as fits
from astropy import wcs as apy_wcs

from forcepho.likelihood import lnlike_multi, make_image
from forcepho.data import PostageStamp
from forcepho import psf as pointspread


__all__ = ["Posterior",
           "negative_lnlike_multi", "chi_vector",
           "numerical_image_gradients",
           "make_stamp", "make_real_stamp"
           ]


class Posterior(object):

    def __init__(self, scene, plans, upper=np.inf, lower=-np.inf, verbose=False):
        self.scene = scene
        self.plans = plans
        self._theta = -99
        self.lower = lower
        self.upper = upper
        self.verbose = verbose
        self.ncall = 0

    def evaluate(self, theta):
        Theta = self.complete_theta(theta)
        if self.verbose:
            print(Theta)
            t = time.time()
        nll, nll_grad = negative_lnlike_multi(Theta, scene=self.scene, plans=self.plans)
        if self.verbose:
            print(time.time() - t)
        self.ncall += 1
        self._lnp = -nll
        self._lnp_grad = -nll_grad
        self._theta = Theta

    def lnprob(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp

    def lnprob_grad(self, Theta):
        if np.any(Theta != self._theta):
            self.evaluate(Theta)
        return self._lnp_grad

    def complete_theta(self, theta):
        return theta

    def check_constrained(self, theta):
        """Method that checks parameter values against constraints.  If they
        are above or below the boundaries, the sign of the momentum is flipped
        and theta is adjusted as if the trajectory had bounced off the
        constraint.

        :param theta:
            The parameter vector

        :returns theta:
            the new theta vector

        :returns sign:
            a vector of multiplicative signs for the momenta

        :returns flag:
            A flag for if the values are still out of bounds.
        """

        #initially no flips
        sign = np.ones_like(theta)
        oob = True #pretend we started out-of-bounds to force at least one check
        #print('theta_in ={0}'.format(theta))
        while oob:
            above = theta > self.upper
            theta[above] = 2*self.upper[above] - theta[above]
            sign[above] *= -1
            below = theta < self.lower
            theta[below] = 2*self.lower[below] - theta[below]
            sign[below] *= -1
            oob = np.any(below | above)
            #print('theta_out ={0}'.format(theta))
        return theta, sign, oob


def negative_lnlike_multi(Theta, scene=None, plans=None, grad=True):
    lnp, lnp_grad = lnlike_multi(Theta, scene=scene, plans=plans)
    if grad:
        return -lnp, -lnp_grad
    else:
        return -lnp

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
        imlo, _ = make_image(scene, stamp, Theta=theta)
        theta[i] += dp
        imhi, _ = make_image(scene, stamp, Theta=theta)
        dI_dp.append((imhi - imlo) / (dp))

    return np.array(dI_dp)

            
def make_stamp(size=(100, 100), fwhm=1.0, psfname=None,
               offset=0., filtername='dummy', oversample=8,
               psfcenter=104):
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

        #oversample, pcenter = 8, 504 - 400  # HAAAACKKK
        answer = pdat[6][2]
        stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=psfcenter)
    else:
        stamp.psf = pointspread.PointSpreadFunction()
        stamp.psf.covariances *= fwhm / 2.355
        
    # --- Add extra information ---
    #stamp.full_header = dict(hdr)
    stamp.filtername = filtername

    return stamp


def make_real_stamp(imname, center=(None, None), size=(None, None),
                    center_type='pixels', psfname=None, fwhm=1.0,
                    oversample=8, psfcenter=104, fix_header=False,
                    psf_ngauss=6, psf_realization=2):
    """Make a postage stamp around the given position using the given image name
    """
    data = fits.getdata(imname)
    hdr = fits.getheader(imname)
    crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
    crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                   [hdr['CD2_1'], hdr['CD2_2']]])

    if fix_header:
        # because hdr PC/CD/CDELT is sometimes wrong.
        hdr["CDELT1"] = hdr['CD1_1'] / hdr["PC1_1"]
        hdr["CDELT2"] = hdr['CD2_2'] / hdr["PC2_2"]

    # Pull slices and transpose to get to an axis order that makes sense to me
    # and corresponds with the wcs keyword ordering
    im = data[0, :, :].T
    err = data[1, :, :].T

    # ---- Extract subarray -----
    center = np.array(center)
    # here we get the center coordinates in pixels (accounting for the transpose above)
    if center_type == 'celestial':
        world = center.copy()
        ast = apy_wcs.WCS(hdr, naxis=2)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2]
    # --- here is much mystery ---
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    crpix_stamp = np.floor(0.5 * size)
    crval_stamp = crpix_stamp + lo
    W = np.eye(2)
    if center_type == 'celestial':
        crval_stamp = ast.wcs_pix2world(crval_stamp[None,:], 0)[0, :2]
        W[0, 0] = np.cos(np.deg2rad(crval_stamp[-1]))


    # --- MAKE STAMP -------

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds]
    stamp.ierr = 1./err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0
    stamp.ierr = stamp.ierr.flatten()

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval_stamp
    stamp.dpix_dsky = np.matmul(np.linalg.inv(CD), W)
    stamp.scale = np.linalg.inv(CD * 3600.0)
    stamp.pixcenter_in_full = center
    stamp.lo = lo
    stamp.CD = CD
    stamp.W = W
    try:
        stamp.wcs = ast
    except:
        pass

    # --- Add the PSF ---
    if psfname is not None:
        import pickle
        with open(psfname, 'rb') as pf:
            pdat = pickle.load(pf)

        answer = pdat[psf_ngauss][psf_realization]
        stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=psfcenter)
    else:
        stamp.psf = pointspread.PointSpreadFunction()
        stamp.psf.covaraniaces *= fwhm/2.355

    # --- Add extra information ---
    stamp.full_header = dict(hdr)
    stamp.filtername = stamp.full_header["FILTER"]

    return stamp


class Result(object):

    def __init__(self):
        self.offsets = None

    
