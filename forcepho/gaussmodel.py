# Some code along the lines of what we will do in C++


import numpy as np

__all__ = ["ImageGaussian", "Star", "Galaxy", "GaussianImageGalaxy",
           "convert_to_gaussians", "get_gaussian_gradients",
           "compute_gaussian"]


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
    """This is a represenation of a point source in terms of Scene (on-sky)
    parameters
    """

    id = 0
    fixed = False
    ngauss = 1
    radii = np.zeros(0)

    # Parameters
    flux = 0.     # flux
    ra = 0.
    dec = 0.
    q = 1.        # axis ratio squared, i.e.  (b/a)^2
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
    """Parameters describing a gaussian galaxy in the celestial plane (i.e. the Scene parameters)
    For each galaxy there are 7 parameters:
      * flux: total flux
      * ra: right ascension (degrees)
      * dec: declination (degrees)
      * q, pa: axis ratio squared and position angle (might be parameterized differently in future)
      * n: sersic index
      * r: half-light radius (arcsec)

    Methods are provided to return the amplitudes and covariance matrices of
    the constituent gaussians, as well as derivatives of the amplitudes with
    respect to sersic index and half light radius.
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
        (dependent on self.n and self.r).  Placeholder code gives them all
        equal amplitudes.
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
        """This just constructs a set of covariance matrices based on the fixed
        radii used in approximating the galaxies.
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return (self.radii**2)[:, None, None] * np.eye(2)


class GaussianImageGalaxy(object):
    """ A list of ImageGaussians corresponding to one galaxy, after image
    distortions, PSF, and in the pixel space.  Like `GaussianGalaxy` in the c++
    code.
    """
    id = 0

    def __init__(self, ngalaxy, npsf, id=None):
        self.id = id
        self.gaussians = np.zeros([ngalaxy, npsf], dtype=object)


def convert_to_gaussians(galaxy, stamp):
    """Takes a set of source parameters into a set of ImagePlaneGaussians,
    including PSF, and keeping track of the dGaussian_dScene.

    :param galaxy:
        A Galaxy() or Star() instance, with the proper parameters.

    :param stamp:
        A PostageStamp() instance, with a valid PointSpreadFunction and
        distortion matrix.

    :returns gig:
       An instance of GaussianImageGalaxy, containing an array of
       ImageGaussians.
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
    """Compute the Jacobian for dphi_i/dtheta_j where phi are the parameters of
    the Image Gaussian and theta are the parameters of the Source in the Scene.

    :param galaxy:
        A source like a Galaxy() or a Star()

    :param stamp:
        An instance of PostageStamp with distortion matrix and valid
        PointSpreadFunction.

    :param gig:
        The `GaussianImageGalaxy` instance that resulted from
        `convert_to_gaussian(galaxy, stamp)`.  This is necessary only because
        the computed Jacobians will be assigned to the `deriv` attribute of
        each `ImageGaussian` in the `GaussianImageGalaxy`.

    :returns gig:
        The same as the input `gig`, but with the computed Jacobians assigned
        to the `deriv` attribute of each of the consitituent `ImageGaussian`s
    """
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
            # each row is a different theta and has
            # dA/dtheta, dx/dtheta, dy/dtheta, dFxx/dtheta, dFyy/dtheta, dFxy/dtheta
            jac = [[dA_dflux, 0, 0, 0, 0, 0],  # d/dFlux
                   [0, D[0, 0], D[1, 0], 0, 0, 0],  # d/dAlpha
                   [0, D[0, 1], D[1, 1], 0, 0, 0],  # d/dDelta
                   [dA_dq, 0, 0, dF_dq[0,0], dF_dq[1,1], dF_dq[1,0]],  # d/dQ
                   [dA_dpa, 0, 0, dF_dpa[0,0], dF_dpa[1,1], dF_dpa[1,0]],  # d/dPA
                   [dA_dsersic, 0, 0, 0, 0, 0],  # d/dSersic
                   [dA_drh, 0, 0, 0, 0, 0]]  # d/dRh
            derivs = np.array(jac)
            gig.gaussians[i, j].derivs = derivs

    return gig


def compute_gaussian(g, xpix, ypix, second_order=True, compute_deriv=True,
                     use_det=False, oversample=False):
    """Calculate the counts and gradient for one pixel, one gaussian.  This
    basically amounts to evalutating the gaussian (plus second order term) and
    the gradients with respect to the 6 input terms.  Like `computeGaussian`
    and `computeGaussianDeriv` in the C++ code.

    :param g: 
        An ImageGaussian() instance, or an object that has the attributes
        `xcen`, `ycen`, `amp`, `fxx`, `fyy` and `fxy`.

    :param xpix:
        The x coordinate of the pixel(s) at which counts and gradient are desired.
        Scalar or ndarray of shape npix.

    :param ypix:
        The y coordinate of the pixel(s) at which counts and gradient are desired.
        Same shape as `xpix`.

    :param second_order: (optional, default: True)
        Whether to use the 2nd order correction to the integral of the gaussian
        within a pixel.

    :param compute_deriv: (optional, default: True)
        Whether to compute the derivatives of the counts with respect to the
        gaussian parameters.

    :param use_det: (otional, default: False)
        Whether to include the determinant of the covariance matrix when
        normalizing the counts and calculating derivatives.

    :param oversample: (optional, default: False)
        If this flag is set, the counts (and derivatives) will be calculated on
        a grid oversampled by a factor of two in each dimension (assuming the
        input xpix and ypix are integers), and then summed at the end to
        produce a more precise measure of the counts in a pixel.
        Not actually working yet.

    :returns counts:
        The counts in each pixel due to this gaussian, same shape as `xpix` and
        `ypix`.

    :returns grad:
        The gradient of the counts with respect to the 6 parameters of the
        gaussian in image coordinates, ndarray of shape (nderiv, npix)
    """

    # All transformations should keep the determinant of the covariance matrix
    # the same, so that it can be folded into the amplitude, making the
    # following uninportant
    # inv_det = fxx*fyy + 2*fxy
    # norm = np.sqrt((inv_det / 2*np.pi))

    if oversample:
        xoff = np.array([0.25, -0.25, -0.25, 0.25])
        yoff = np.array([0.25, 0.25, -0.25, -0.25])
        xp = np.array(xpix)[..., None] + xoff[..., :]
        yp = np.array(ypix)[..., None] + yoff[..., :]
    else:
        xp = xpix
        yp = ypix

    # --- Calculate useful variables ---
    dx = xp - g.xcen
    dy = yp - g.ycen
    vx = g.fxx * dx + g.fxy * dy
    vy = g.fyy * dy + g.fxy * dx
    Gp = np.exp(-0.5 * (dx*vx + dy*vy))
    # G = np.exp(-0.5 * (dx*dx*fxx + 2*dx*dy*fxy + dy*dy*fyy))
    H = 1.0
    root_det = 1.0

    # --- Calculate counts ---
    if second_order:
        H = 1 + (vx*vx + vy*vy - g.fxx - g.fyy) / 24.
    if use_det:
        root_det = np.sqrt(g.fxx * g.fyy - g.fxy * g.fxy)
    C = g.amp * Gp * H * root_det

    # --- Calculate derivatives ---
    if compute_deriv:
        dC_dA = C / g.amp
        dC_dx = C*vx
        dC_dy = C*vy
        dC_dfx = -0.5*C*dx*dx
        dC_dfy = -0.5*C*dy*dy
        dC_dfxy = -1.0*C*dx*dy

    if compute_deriv and second_order:
        c_h = C / H
        dC_dx -= c_h * (g.fxx*vx + g.fxy*vy) / 12.
        dC_dy -= c_h * (g.fyy*vy + g.fxy*vx) / 12.
        dC_dfx -= c_h * (1. - 2.*dx*vx) / 24.
        dC_dfy -= c_h * (1. - 2.*dy*vy) / 24.
        dC_dfxy += c_h * (dy*vx + dx*vy) / 12.

    if compute_deriv and use_det:
        c_d = C / (root_det * root_det)
        dC_dfx += 0.5 * c_d * g.fyy
        dC_dfy += 0.5 * c_d * g.fxx
        dC_dfxy -= c_d * g.fxy

    C = np.array(C)
    gradients = np.zeros(1)
    if compute_deriv:
        gradients = np.array([dC_dA, dC_dx, dC_dy, dC_dfx, dC_dfy, dC_dfxy])

    if oversample:
        C = C.sum(axis=-1) / 4.0
        gradients = gradients.sum(axis=-1) / 4.0

    if compute_deriv:
        return C, gradients
    else:
        return C


def scale_matrix(q):
    #return np.array([[q**(-0.5), 0],
    #                [0, q**(0.5)]])
    return np.array([[1./q, 0],  #use q=(b/a)**2
                    [0, q]])


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def scale_matrix_deriv(q):
    #return np.array([[-0.5 * q**(-1.5), 0],
    #                [0, 0.5 * q**(-0.5)]])
    return np.array([[-1./q**2, 0], # use q=(b/a)**2
                    [0, 1]])


def rotation_matrix_deriv(theta):
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]])
