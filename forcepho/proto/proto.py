# Some code along the lines of what we will do in C++

import numpy as np


class ImageGaussian(object):
    """This is description of one Gaussian, in pixel locations, as well as the
    derivatives of this Gaussian wrt to the Scene model parameters.

    The gaussian is described by 6 parameters
    * Amplitude A
    * Inverse Covariance matrix [[Fxx, Fxy], [Fxy, Fyy]]
    * center xcen, ycen

    In the Scene space there are 7 parameters, so the dgauss_dscene matrix is 6
    x 7 (mostly zeros)
    """
    A = 0.
    xcen = 0.
    ycen = 0.
    Fxx = 0.
    Fxy = 0.
    Fyy = 0.
    #float dGaussian_dScene[NDERIV];


class Galaxy(object):
    """Parameters describing a galaxy in the celestial plane (i.e. the Scene parameters)
    For each galaxy there are 7 parameters.
    * psi: total flux
    * ra: right ascension (degrees)
    * dec: declination (degrees)
    * q, phi: axis ratio and position angle (might be parameterized differently)
    * n: sersic index
    * r: half-light radius (arcsec)
    
    """
    id = 0
    psi = 0. # flux
    ra = 0.
    dec = 0.
    q = 0.  # axis ratio
    phi = 0.  # position angle (N of E)
    n = 0.  # sersic index
    r = 0.  # half-light radius (arcsec)
    fixed = False
    ngauss = 10
    radii = np.arange(ngauss)

    @property
    def amplitudes(self):
        """Code here for getting amplitudes from a splined look-up table
        (dependent on self.n and self.r)
        """
        return amps

    @property
    def amp_derivs(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss x nscene_params, only two are nonzero (n and r)
        return amp_derivs

    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the radii
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return covars


class Scene(object):
    """A collection of fixed and free galaxies describing the scene
    """
    
class GaussianImageGalaxy(object):
    """ A list of gaussians corresponding to one galaxy, after image
    distortions, PSF, and in the pixel space.  Like `GaussianGalaxy` in the c++
    code.
    """
    id = 0


class PointSpreadFunction(object):
    """Gaussian Mixture approximation to a PSF.
    """

    ngauss = 1
    covariances = np.array(self.ngauss * [[1.,0.], [0., 1.]])
    means = np.zeros([self.ngauss, 2])
    amplitudes = np.ones(self.ngauss)
    
    
class PostageStamp(object):
    """A list of pixel values and locations, along with the distortion matrix,
    astrometry, and PSF.

    The astrometry is such that 
    :math:`x = D \, (c - c_0) + x_0`
    where x_0 are the reference pixel coordinates, c_0 are the reference pixel
    *celestial* coordinates (in degrees of RA and Dec), and D is the
    distortion_matrix.
    """

    id = 1

    # Size of the stamp
    nx = 100
    ny = 100

    # The distortion matrix D
    distortion = np.ones([2,2])
    # The sky coordinates of the reference pixel
    crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    crpix = np.zeros([2])

    psf = PointSpreadFunction()

    def sky_to_pix(self, sky):
        pix = np.dot(self.distortion, sky - self.crval) + crpix
        return pix

    def pix_to_sky(self, pix):
        pass


def convert_to_gaussians(galaxy, stamp):
    """Takes the galaxy parameters into a set of ImagePlaneGaussians,
    including PSF, and keeping track of the dGaussian_dScene
    """
    # Get the transformation matrix
    D = stamp.distortion
    R = rotation_matrix(galaxy.phi)
    S = scale_matrix(galaxy.q)
    T = np.dot(D, np.dot(R, S))

    # And its deriatives with respect to scene parameters
    dS_dq = scale_matrix_deriv(galaxy.q)
    dR_dphi = rotation_matrix_deriv(galaxy.phi)
    #dT_dScene = np.zeros([7, 2, 2])
    dT_dScene[3] = np.dot(D, np.dot(R, dS_dq))
    dT_dScene[4] = np.dot(D, np.dot(dR_dphi, S))

    # get galaxy component means, covariances, and amplitudes in the pixel space
    gcovar = np.matmul(T, np.matmul(galaxy.covariances, T.T))
    gamps = galaxy.amplitudes
    gmean = stamp.sky_to_pix([galaxy.ra, galaxy.dec])

    # convolve with the PSF for this stamp, yield Ns x Np sets of means, covariances, and amplitudes.
    #covar = gcovar[:, None, :, :] + stamp.psf_covar[None, :, :, :]
    #amps = gamps[:, None] * stamp.psf_amplitudes[None, :]
    #means = gmeans[:, None, :] + stamp.psf_means[None, :, :]

    gig = GaussianImageGalaxy(galaxy.ngauss, stamp.psf.ngauss,
                              id=(galaxy.id, stamp.id))
    for i in range(galaxy.ngauss):
        # gcovar = np.matmul(T, np.matmul(galaxy.covariances[i], T.T))
        for j in range(stamp.psf.ngauss):
            gaussid = (galaxy.id, stamp.id, i, j)
            # convolve the jth Galaxy component with the ith PSF component
            covar = gcovar[i] + stamp.psf.covariances[j]
            f = np.linalg.inv(covar)
            fxx = f[0, 0]
            fxy = f[1, 0]
            fyy = f[1, 1]
            xcen, ycen = gmean + stamp.psf.means[j]
            a = gamps[i] * stamp.psf.amplitudes[j]
            # adjust a for the determinant of F
            # blah
            # Now get derivatives
            dSigma_dScene = (np.matmul(dT_dScene, np.matmul(gcovar, T.T)) +
                             np.matmul(T, np.matmul(gcovar, np.transpose(dT_dScene, [0,2,1])))
                             )
            df_dScene = np.matmul(f, np.matmul(dSigma_dScene, f))
            
            # And add to list of gaussians
            gig.gaussians[i, j] = ImageGaussian(a, xcen, ycen, fxx, fxy, fyy,
                                                id=gaussid, derivs=None))

    return gig


def get_gaussian_gradients(galaxy, stamp, gig):
    # Get the transformation matrix
    D = stamp.distortion
    R = rotation_matrix(galaxy.phi)
    S = scale_matrix(galaxy.q)
    T = np.dot(D, np.dot(R, S))

    # And its deriatives with respect to scene parameters
    dS_dq = scale_matrix_deriv(galaxy.q)
    dR_dphi = rotation_matrix_deriv(galaxy.phi)
    #dT_dScene = np.zeros([7, 2, 2])
    dT_dq = np.dot(D, np.dot(R, dS_dq))
    dT_dphi = np.dot(D, np.dot(dR_dphi, S))

    # get galaxy component means, covariances, and amplitudes in the pixel space
    gcovar = np.matmul(T, np.matmul(galaxy.covariances, T.T))
    gamps = galaxy.amplitudes
    gmean = stamp.sky_to_pix([galaxy.ra, galaxy.dec])

    
    for i in range(galaxy.ngauss):
        # gcovar = np.matmul(T, np.matmul(galaxy.covariances[i], T.T))
        for j in range(stamp.psf.ngauss):
            gaussid = (galaxy.id, stamp.id, i, j)
            # convolve the jth Galaxy component with the ith PSF component
            covar = gcovar[i] + stamp.psf.covariances[j]
            F = np.linalg.inv(covar)
            detF
            Fxx = f[0, 0]
            Fxy = f[1, 0]
            Fyy = f[1, 1]
            xcen, ycen = gmean + stamp.psf.means[j]
            a = gamps[i] * stamp.psf.amplitudes[j]
            # adjust a for the determinant of F
            # blah
            # Now get derivatives
            #dSigma_dq = (np.matmul(dT_dScene, np.matmul(gcovar, T.T)) +
            #                 np.matmul(T, np.matmul(gcovar, np.transpose(dT_dScene, [0,2,1])))
            #                 )
            
            dSigma_dphi = 
            dF_dq = np.matmul(f, np.matmul(dSigma_dScene, f))
            
            # And add to list of gaussians
            gig.gaussians[i, j] = ImageGaussian(a, xcen, ycen, fxx, fxy, fyy,
                                                id=gaussid, derivs=None))



def compute_gaussian(g, xpix, ypix, second_order=True, compute_deriv=True):
    """Calculate the counts and gradient for one pixel, one gaussian.  This
    basically amounts to evalutating the gaussian (plus second order term) and
    the gradients with respect to the 6 input terms.  Like `computeGaussian`
    and `computeGaussianDeriv` in the C++ code.

    :param g: 
        A Gaussian() instance

    :param xpix:
        The x coordinate of the pixel(s) at which counts and gradient are desired.

    :param ypix:
        The y coordinate of the pixel(s) at which counts and gradient are desired.

    :returns counts:
        The counts in this pixel due to this gaussian

    :returns grad:
        The gradient of the counts with respect to the 6 parameters of the
        gaussian in image coordinates
    """

    # All transformations should keep the determinant of the covariance matrix
    # the same, so that it can be folded into the amplitude, making the
    # following uninportant
    # inv_det = fxx*fyy + 2*fxy
    # norm = np.sqrt((inv_det / 2*np.pi))

    dx = xpix - g.xcen
    dy = ypix - g.ycen
    vx = g.Fxx * dx + g.Fxy * dy
    vy = g.Fyy * dy + g.Fxy * dx
    #G = np.exp(-0.5 * (dx*dx*fxx + 2*dx*dy*fxy + dy*dy*fyy))
    Gp = np.exp(-0.5 * (dx*vx + dy*vy))
 
    if second_order:
        H = 1 + (vx*vx + vy*vy - g.Fxx - g.Fyy) / 24.
    else:
        H = 1.0

    c_h = A * Gp
    C = c_h * H

    if not compute_deriv:
        return np.array(C)

    dC_dA = Gp * H
    dC_dx = C*vx - second_order * c_h * 2./24. * (g.Fxx*vx + g.Fxy*vy) 
    dC_dy = C*vy - second_order * c_h * 2./24. * (g.Fyy*vy + g.Fxy*vx)
    dC_dfx = -0.5*C*dx*dx - second_order * c_h * (1. + 2.*dx*vx) / 24.
    dC_dfy = -0.5*C*dy*dy - second_order * c_h * (1. + 2.*dy*vy) / 24.
    dC_dfxy = -1.0*C*dx*dy - second_order * c_h * (1. + 2.*dy*vy) / 24.

    return np.array(C), np.array([dC_dA, dC_dx, dC_dy, dC_dfx, dC_dfy, dC_dfxy]).T


def scale_matrix(q):
        return np.array([[1/q, 0],
                        [0, q]])


def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])


def scale_matrix_deriv(q):
        return np.array([[-1/q**2, 0],
                        [0, 1]])


def rotation_matrix_deriv(theta):
        return np.array([[-np.sin(theta), -np.cos(theta)],
                         [np.cos(theta), -np.sin(theta)]])
