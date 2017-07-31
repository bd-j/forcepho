# Some code along the lines of what we will do in C++


import numpy as np

__all__ = ["ImageGaussian", "Star", "Galaxy", "GaussianImageGalaxy",
           "PostageStamp", "PointSpreadFunction",
           "convert_to_gaussians", "get_gaussian_gradients", "compute_gaussian"]



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
    amp = 0.
    xcen = 0.
    ycen = 0.
    fxx = 0.
    fxy = 0.
    fyy = 0.

    # derivs = [dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh, D.flatten(), dF_dq.flat[inds], dF_dpa.flat[inds]]
    derivs = None  # this is the dGaussian_dScene Jacobian matrix, possibly in a sparse format
    #float dGaussian_dScene[NDERIV];
    

class Star(object):

    id = 0
    fixed = False
    ngauss = 1
    radii = np.zeros(0)

    # Parameters
    flux = 0.     # flux
    ra = 0.
    dec = 0.
    q = 1.        # axis ratio
    pa = 0.       # postion angle (N of E)
    sersic = 0.   # sersic index
    rh = 0.       # half light radius

    def __init__(self):
        pass
    
    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the radii
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return np.zeros([1, 2, 2])

    @property
    def amplitudes(self):
        """
        """
        return np.ones(1)

    @property
    def damplitude_dsersic(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/dn
        return np.zeros(1)

    @property
    def damplitude_drh(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/dr
        return np.zeros(1)

    
class Galaxy(object):
    """Parameters describing a galaxy in the celestial plane (i.e. the Scene parameters)
    For each galaxy there are 7 parameters.
    * flux: total flux
    * ra: right ascension (degrees)
    * dec: declination (degrees)
    * q, pa: axis ratio and position angle (might be parameterized differently)
    * n: sersic index
    * r: half-light radius (arcsec)
    
    """
    id = 0
    fixed = False
    ngauss = 10
    radii = np.arange(ngauss)

    # Parameters
    flux = 0.     # flux
    ra = 0.
    dec = 0.
    q = 1.        # axis ratio
    pa = 0.       # postion angle (N of E), in radians
    sersic = 0.   # sersic index
    rh = 0.       # half light radius

    @property
    def amplitudes(self):
        """Code here for getting amplitudes from a splined look-up table
        (dependent on self.n and self.r)
        """
        return np.ones(self.ngauss) / (self.ngauss * 1.0)

    @property
    def damplitude_dsersic(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/dsersic
        return np.zeros(self.ngauss)

    @property
    def damplitude_drh(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/drh
        return np.zeros(self.ngauss)

    
    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the radii
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return (self.radii**2)[:, None, None] * np.eye(2)


class GaussianImageGalaxy(object):
    """ A list of gaussians corresponding to one galaxy, after image
    distortions, PSF, and in the pixel space.  Like `GaussianGalaxy` in the c++
    code.
    """
    id = 0

    def __init__(self, ngalaxy, npsf, id=None):
        self.id = id
        self.gaussians = np.zeros([ngalaxy, npsf], dtype=object)


class PointSpreadFunction(object):
    """Gaussian Mixture approximation to a PSF.
    """

    def __init__(self):
        self.ngauss = 1
        self.covariances = np.array(self.ngauss * [[[1.,0.], [0., 1.]]])
        self.means = np.zeros([self.ngauss, 2])
        self.amplitudes = np.ones(self.ngauss)

    
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
    npix = 100 * 100

    # The distortion matrix D
    distortion = np.eye(2)
    # The sky coordinates of the reference pixel
    crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    crpix = np.zeros([2])

    # The point spread function
    psf = PointSpreadFunction()

    # The pixel values and residuals
    pixel_value = np.zeros([nx, ny])
    residual = np.zeros([nx, ny])
    ierr = np.zeros([nx, ny])

    def sky_to_pix(self, sky):
        pix = np.dot(self.distortion, sky - self.crval) + self.crpix
        return pix

    def pix_to_sky(self, pix):
        pass


def convert_to_gaussians(galaxy, stamp):
    """Takes the galaxy parameters into a set of ImagePlaneGaussians,
    including PSF, and keeping track of the dGaussian_dScene
    """
    # Get the transformation matrix
    D = stamp.distortion
    R = rotation_matrix(galaxy.pa)
    S = scale_matrix(galaxy.q)
    T = np.dot(D, np.dot(R, S))

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
            gauss = ImageGaussian()
            gauss.id = (galaxy.id, stamp.id, i, j)
            
            # Convolve the jth Galaxy component with the ith PSF component

            # Covariance matrix
            covar = gcovar[i] + stamp.psf.covariances[j]
            f = np.linalg.inv(covar)
            gauss.fxx = f[0, 0]
            gauss.fxy = f[1, 0]
            gauss.fyy = f[1, 1]

            # Now get centers and amplitudes
            # TODO: Add gain/conversion from stamp to go from physical flux to counts.
            gauss.xcen, gauss.ycen = gmean + stamp.psf.means[j]
            am, al = galaxy.amplitudes[i], stamp.psf.amplitudes[j]
            gauss.amp = galaxy.flux * am * al * np.linalg.det(f)**(0.5) / (2 * np.pi)

            # And add to list of gaussians
            gig.gaussians[i, j] = gauss

    return gig


def get_gaussian_gradients(galaxy, stamp, gig):
    # Get the transformation matrix
    D = stamp.distortion
    R = rotation_matrix(galaxy.pa)
    S = scale_matrix(galaxy.q)
    T = np.dot(D, np.dot(R, S))

    # And its deriatives with respect to scene parameters
    dS_dq = scale_matrix_deriv(galaxy.q)
    dR_dpa = rotation_matrix_deriv(galaxy.pa)
    dT_dq = np.dot(D, np.dot(R, dS_dq))
    dT_dpa = np.dot(D, np.dot(dR_dpa, S))

    for i in range(galaxy.ngauss):
        gcovar = galaxy.covariances[i]
        gcovar_im = np.matmul(T, np.matmul(gcovar, T.T))
        for j in range(stamp.psf.ngauss):
            #gaussid = (galaxy.id, stamp.id, i, j)

            # convolve the jth Galaxy component with the ith PSF component
            Sigma = gcovar_im + stamp.psf.covariances[j]
            F = np.linalg.inv(Sigma)
            detF = np.linalg.det(F)
            am, al = galaxy.amplitudes[i], stamp.psf.amplitudes[j]
            K = galaxy.flux * am * al * detF**(0.5) / (2 * np.pi)
            
            # Now get derivatives
            # Of F
            dSigma_dq = (np.matmul(T, np.matmul(gcovar, dT_dq.T)) +
                         np.matmul(dT_dq, np.matmul(gcovar, T.T)))
            dSigma_dpa = (np.matmul(T, np.matmul(gcovar, dT_dpa.T)) +
                          np.matmul(dT_dpa, np.matmul(gcovar, T.T)))
            dF_dq = -np.matmul(F, np.matmul(dSigma_dq, F))  # 3
            dF_dpa = -np.matmul(F, np.matmul(dSigma_dpa, F))  # 3
            ddetF_dq = detF * np.trace(np.matmul(Sigma, dF_dq))
            ddetF_dpa = detF * np.trace(np.matmul(Sigma, dF_dpa))
            # of Amplitude
            # TODO: Add gain/conversion from stamp to go from physical flux to counts
            dA_dq = K / (2 * detF) * ddetF_dq  # 1
            dA_dpa = K / (2 * detF) * ddetF_dpa  # 1
            dA_dflux = K / galaxy.flux # 1
            dA_dsersic = K / am * galaxy.damplitude_dsersic[i] # 1
            dA_drh = K / am * galaxy.damplitude_drh[i] # 1

            # Add derivatives to gaussians
            # As a list of just nonzero
            inds = [0, 1, 3] # slice into a flattened symmetric matrix to get unique components
            derivs = ([dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh] + 
                       D.flatten().tolist() +
                       dF_dq.flat[inds].tolist() +
                       dF_dpa.flat[inds].tolist())
            # as the 7 x 6 jacobian matrix
            jac = [[dA_dflux, 0, 0, 0, 0, 0],  # d/dFlux
                   [0, D[0, 0], D[1, 0], 0, 0, 0],  # d/dAlpha
                   [0, D[0, 1], D[1, 1], 0, 0, 0],  # d/dDelta
                   [dA_dq, 0, 0, dF_dq[0,0], dF_dq[1,0], dF_dq[1,1]],  # d/dQ
                   [dA_dpa, 0, 0, dF_dpa[0,0], dF_dpa[1,0], dF_dpa[1,1]],  # d/dPA
                   [dA_dsersic, 0, 0, 0, 0, 0],  # d/dSersic
                   [dA_drh, 0, 0, 0, 0, 0]]  # d/dRh
            derivs = np.array(jac)
            gig.gaussians[i, j].derivs = derivs

    return gig


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
        gaussian in image coordinates, ndarray of shape (nderiv, npix)
    """

    # All transformations should keep the determinant of the covariance matrix
    # the same, so that it can be folded into the amplitude, making the
    # following uninportant
    # inv_det = fxx*fyy + 2*fxy
    # norm = np.sqrt((inv_det / 2*np.pi))

    dx = xpix - g.xcen
    dy = ypix - g.ycen
    vx = g.fxx * dx + g.fxy * dy
    vy = g.fyy * dy + g.fxy * dx
    #G = np.exp(-0.5 * (dx*dx*fxx + 2*dx*dy*fxy + dy*dy*fyy))
    Gp = np.exp(-0.5 * (dx*vx + dy*vy))
 
    if second_order:
        H = 1 + (vx*vx + vy*vy - g.fxx - g.fyy) / 24.
    else:
        H = 1.0

    c_h = g.amp * Gp
    C = c_h * H

    if not compute_deriv:
        return np.array(C)

    dC_dA = Gp * H
    dC_dx = C*vx - second_order * c_h * 2./24. * (g.fxx*vx + g.fxy*vy) 
    dC_dy = C*vy - second_order * c_h * 2./24. * (g.fyy*vy + g.fxy*vx)
    dC_dfx = -0.5*C*dx*dx - second_order * c_h * (1. + 2.*dx*vx) / 24.
    dC_dfy = -0.5*C*dy*dy - second_order * c_h * (1. + 2.*dy*vy) / 24.
    dC_dfxy = -1.0*C*dx*dy - second_order * c_h * (1. + 2.*dy*vy) / 24.

    return np.array(C), np.array([dC_dA, dC_dx, dC_dy, dC_dfx, dC_dfy, dC_dfxy])


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
