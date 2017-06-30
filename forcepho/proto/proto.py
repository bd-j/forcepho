# Some code along the lines of what we will do in C++

import numpy as np


class Gaussian(object):
    """This is description of one Gaussian, in pixel locations, as well as the
    derivatives of this Gaussian wrt to the Scene model parameters.

    The gaussian is described by 
    * Amplitude A
    * Inverse Covariance matrix [[Fxx, Fxy], [Fxy, Fyy]]
    * center xcen, ycen

    """
    A = 0.
    xcen = 0.
    ycen = 0.
    Fxx = 0.
    Fxy = 0.
    Fyy = 0.
    #float dGaussian_dScene[NDERIV];


class Galaxy(object):
    """Parameters describing a galaxy in the celestial plane.
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


    def amplitudes(self):
        return amps

    def amp_derivs(self):
        # ngauss x nscene_params, only two are nonzero (n and r)
        return amp_derivs

    def covariances(self):
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return covars

    
    
class Scene(object):
    """A collection of fixed and free galaxies describing the scene
    """
    
class GaussianImageGalaxy(object):
    """ A list of gaussians corresponding to one galaxy, after image
    distortions, PSF, and in the pixel space.
    """
    id = 0


class PostageStamp(object):
    """A list of pixel values and locations, along with the distortion matrix,
    astrometry, and PSF.

    The astrometry is such that 
    :math:`x = D \, (c - c_0) + x_0`
    where x_0 are the reference pixel coordinates, c_0 are the reference pixel
    *celestial* coordinates (in degrees of RA and Dec), and D is the
    distortion_matrix.
    """
    # Size of the stamp
    nx = 100
    ny = 100

    # The distortion matrix D
    distortion = np.ones([2,2])
    # The sky coordinates of the reference pixel
    crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    crpix = np.zeros([2])

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
    # And it's deriative with respect to scene parameters
    # blah

    # get galaxy component means, covariances, and amplitudes in the pixel space
    gcovar = np.matmul(T, np.matmul(galaxy.covariances, T.T))
    gamps = galaxy.amplitudes(galaxy.n, galaxy.r)
    gmean = stamp.sky_to_pix([galaxy.ra, galaxy.dec])

    # convolve with the PSF for this stamp, yield Ns x Np sets of means, covariances, and amplitudes.
    #covar = gcovar[:, None, :, :] + stamp.psf_covar[None, :, :, :]
    #amps = gamps[:, None] * stamp.psf_amplitudes[None, :]
    #means = gmeans[:, None, :] + stamp.psf_means[None, :, :]

    gig = GaussianImageGalaxy(galaxy.ngauss, stamp.psf.ngauss,
                              id=(galaxy.id, stamp.id))
    for i in range(galaxy.ngauss):
        # gcovar = np.matmul(T, np.matmul(np.eye(self.covariances[i]), T.T))
        for j in range(stamp.psf.ngauss):
            gaussid = (galaxy.id, stamp.id, i, j)
            # convolve the jth Galaxy component with the ith PSF component
            covar = gcovar[i] + stamp.psf.covariances[j]
            fxx = covar[0, 0]
            fyy = covar[1, 0]
            fyy = covar[1, 1]
            xcen, ycen = gmean + stamp.psf.means[j]
            a = gamps[i] * stamp.psf.amplitudes[j]
            # adjust a for the determinant of F
            # blah
            # Now get derivatives
            # blah

            # And add to list of gaussians
            gig.gaussians[i, j] = Gaussian(a, xcen, ycen, fxx, fxy, fyy,
                                           id=gaussid, derivs=None))

    return gig


def compute_gaussian(g, xpix, ypix, second_order=True, compute_deriv=True):
    """Calculate the counts and gradient for one pixel, one gaussian.  This
    basically amounts to evalutating the gaussian (plus second order term) and
    the gradients with respect to the 6 input terms.

    :param g: 
        A Gaussian() instance

    :param xpix:
        The x coordinate of the pixel(s) at which counts and gradient are desired.

    :param ypix:
        The y coordinate of the pixel(s) at which counts and gradient are desired.

    :returns counts:
        The counts in this pixel due to this gaussian

    :returns grad:
        The gradient of the counts with respect to the 6 parameters
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

    return np.array(C), np.array([dC_dx, dC_dy, dC_dfx, dC_dfy, dC_dfxy, dC_dA]).T


def counts_pg(pixel, gaussian, jacobian=None):
    """Apply the Jacobian transformation to the gradients to bring them from
    the space of the gaussian parameters (phi) to the source parameters (theta)
    
    :param pixel:
        The pixel center, shape (2, Npix)

    :param gaussian:
        The gaussian parameters, shape (Nphi,)

    :param optional:
        The jacobian of the transformation from gaussian parameters phi to
        source parameters theta, shape (Nphi, Ntheta)
    """
    dx = pixel[0] - gaussian[0]  # (Npix,)
    dy = pixel[1] - gaussian[1]  # (Npix,)
    counts, grad_native = counts_pg_native(dx, dy, *gaussian[2:])  # (Npix,), (Npix, Nphi)
    if jacobian is None:
        return counts, grad_native
    else:
        grad_source += np.dot(grad_native, jacobian)  # (Npix, Ntheta)
        return counts, grad_source
    # Sparse matrix multiply instead of np.dot
    #for (m,n), j in jacobian.items():
    #    grad_source[m] += j * grad[n]


def counts_p(pixels, source_params):
    """Add all gaussians (and gradients thereof) for a given source to a given pixel

    :param pixels:
        Pixel centers, array of shape (2, Npix)

    :param source:
        Source parameters, dictionary of length (Ntheta,) with Ntheta usually 7
    """
    
    # These could be done with a `source` object with `as_mixture()` and
    # `mixture_jacobian()` methods
    source.update(**source_params)
    gaussians = source.as_mixture()  # shape (Ng, Nphi)
    jacobians = source.mixture_jacobian() # shape (Ng, Nphi, Ntheta)

    # allocate output
    image = np.zeros(pixels.shape[-1])
    gradient = np.zeros(pixels.shape[-1], jacobians.shape[-1])

    for g, gaussian in enumerate(gaussians):
        # for p, mu in enumerate(pixels.T):
        c, grad = counts_pg(pixels, gaussian, jacobian=jacobians[g])
        image += c
        gradient += grad

    return image, gradient


class Source(object):

    def __init__(self, xs, ys, q, pa, n, rh, flux):
        self.params = {}

    def update(self, **params):
        for k, v in params.items():
            self.params[k] = v
        self.dirtiness = 1

    def as_mixture(self):
        """Calculate the parameters of a gaussian mixture from a given set of
        source parameters.  This should be a method of a `Source` class.
        """
        gaussians = np.empty([ng, nphi])
        return gaussians

    def mixture_jacobian(self, sparse=False):
        """Calculate the jacobian matrices of the transformation from source to
        gaussian parameters.  This should be a method of a `Source` class.
        Also it should allow for the return of sparse matrices
        """
        if self.dirtiness > 0:
            self._jacobian = np.empty([ng, nphi, ntheta])
            self.dirtiness = 0

        return self._jacobian



def scale_matrix(q):
        return np.array([[1/q, 0],
                        [0, q]])


def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
