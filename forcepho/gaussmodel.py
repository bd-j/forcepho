# Some code along the lines of what we will do in C++


import numpy as np


__all__ = ["ImageGaussian", "GaussianImageGalaxy",
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


class GaussianImageGalaxy(object):
    """ A list of ImageGaussians corresponding to one galaxy, after image
    scalings, PSF, and in the pixel space.  Like `GaussianGalaxy` in the c++
    code.
    """
    id = 0

    def __init__(self, ngalaxy, npsf, id=None):
        self.id = id
        self.gaussians = np.zeros([ngalaxy, npsf], dtype=object)


def convert_to_gaussians(source, stamp, compute_deriv=False):
    """Takes a set of source parameters into a set of ImagePlaneGaussians,
    including PSF, and keeping track of the dGaussian_dScene.

    :param galaxy:
        A Galaxy() or Star() instance, with the proper parameters.

    :param stamp:
        A PostageStamp() instance, with a valid PointSpreadFunction and
        scale matrix.

    :returns gig:
       An instance of GaussianImageGalaxy, containing an array of
       ImageGaussians.
    """
    # Get the transformation matrix
    D = stamp.scale  # pix/arcsec
    R = rotation_matrix(source.pa)
    S = scale_matrix(source.q)
    T = np.dot(D, np.dot(R, S))

    # get source component means, covariances, and amplitudes in the pixel space
    scovar = np.matmul(T, np.matmul(source.covariances, T.T))
    samps = source.amplitudes
    smean = stamp.sky_to_pix([source.ra, source.dec])
    flux = np.atleast_1d(source.flux)
    if len(flux) == 1:
        flux = flux[0]
    else:
        flux = flux[source.filter_index(stamp.filtername)]

    # Convert flux to counts
    flux *= stamp.photocounts

    # get PSF component means and covariances in the pixel space
    if stamp.psf.units == 'arcsec':
        pcovar = np.matmul(D, np.matmul(stamp.psf.covariances, D.T))
        pmean = np.matmul(D, stamp.psf.means)
        # FIXME need to adjust amplitudes to still sum to one?
        pamps = stamp.psf.amplitudes
    elif stamp.psf.units == 'pixels':
        pcovar = stamp.psf.covariances
        pmeans = stamp.psf.means
        pamps = stamp.psf.amplitudes

    # convolve with the PSF for this stamp, yield Ns x Np sets of means, covariances, and amplitudes.
    #covar = gcovar[:, None, :, :] + pcovar[None, :, :, :]
    #amps = gamps[:, None] * pamps[None, :]
    #means = gmean[None, :] + pmeans[None, :, :]

    gig = GaussianImageGalaxy(source.ngauss, stamp.psf.ngauss,
                              id=(source.id, stamp.id))
    for i in range(source.ngauss):
        # scovar = np.matmul(T, np.matmul(source.covariances[i], T.T))
        for j in range(stamp.psf.ngauss):
            gauss = ImageGaussian()
            gauss.id = (source.id, stamp.id, i, j)

            # Convolve the jth Source component with the ith PSF component

            # Covariance matrix
            covar = scovar[i] + pcovar[j]
            f = np.linalg.inv(covar)
            gauss.fxx = f[0, 0]
            gauss.fxy = f[1, 0]
            gauss.fyy = f[1, 1]

            # Now get centers and amplitudes
            gauss.xcen, gauss.ycen = smean + pmeans[j]
            am, al = samps[i], pamps[j]
            gauss.amp = flux * am * al * np.linalg.det(f)**(0.5) / (2 * np.pi)

            # And add to list of gaussians
            gig.gaussians[i, j] = gauss

    if compute_deriv:
        gig = get_gaussian_gradients(source, stamp, gig)

    return gig


def get_gaussian_gradients(source, stamp, gig):
    """Compute the Jacobian for dphi_i/dtheta_j where phi are the parameters of
    the Image Gaussian and theta are the parameters of the Source in the Scene.

    :param source:
        A source like a Galaxy() or a Star()

    :param stamp:
        An instance of PostageStamp with scale matrix and valid
        PointSpreadFunction.

    :param gig:
        The `GaussianImageGalaxy` instance that resulted from
        `convert_to_gaussian(source, stamp)`.  This is necessary only because
        the computed Jacobians will be assigned to the `deriv` attribute of
        each `ImageGaussian` in the `GaussianImageGalaxy`.

    :returns gig:
        The same as the input `gig`, but with the computed Jacobians assigned
        to the `deriv` attribute of each of the consitituent `ImageGaussian`s
    """
    # Get the transformation matrix and other conversions
    D = stamp.scale  # pix/arcsec
    R = rotation_matrix(source.pa)
    S = scale_matrix(source.q)
    T = np.dot(D, np.dot(R, S))
    CW = stamp.dpix_dsky  #dpix/dra, dpix/ddec
    G = stamp.photocounts  # physical to counts

    # And its derivatives with respect to scene parameters
    dS_dq = scale_matrix_deriv(source.q)
    dR_dpa = rotation_matrix_deriv(source.pa)
    dT_dq = np.dot(D, np.dot(R, dS_dq))
    dT_dpa = np.dot(D, np.dot(dR_dpa, S))

    # get source spline and derivatives
    scovars = source.covariances
    samps = source.amplitudes
    da_dn = source.damplitude_dsersic
    da_dr = source.damplitude_drh

    # pull the correct flux from the multiband array
    flux = np.atleast_1d(source.flux)
    if len(flux) == 1:
        flux = flux[0]
    else:
        flux = flux[source.filter_index(stamp.filtername)]

    # get PSF component means and covariances in the pixel space
    if stamp.psf.units == 'arcsec':
        pcovar = np.matmul(D, np.matmul(stamp.psf.covariances, D.T))
        pmean = np.matmul(D, stamp.psf.means)
        # FIXME need to adjust amplitudes to still sum to one?
        pamps = stamp.psf.amplitudes
    elif stamp.psf.units == 'pixels':
        pcovar = stamp.psf.covariances
        pmeans = stamp.psf.means
        pamps = stamp.psf.amplitudes

    for i in range(source.ngauss):
        scovar = scovars[i]
        scovar_im = np.matmul(T, np.matmul(scovar, T.T))
        for j in range(stamp.psf.ngauss):
            #gaussid = (source.id, stamp.id, i, j)

            # convolve the jth Source component with the ith PSF component
            Sigma = scovar_im + pcovar[j]
            F = np.linalg.inv(Sigma)
            detF = np.linalg.det(F)
            am, al = samps[i], pamps[j]
            K = flux * G * am * al * detF**(0.5) / (2 * np.pi)

            # Now get derivatives
            # Of F
            dSigma_dq = (np.matmul(T, np.matmul(scovar, dT_dq.T)) +
                         np.matmul(dT_dq, np.matmul(scovar, T.T)))
            dSigma_dpa = (np.matmul(T, np.matmul(scovar, dT_dpa.T)) +
                          np.matmul(dT_dpa, np.matmul(scovar, T.T)))
            dF_dq = -np.matmul(F, np.matmul(dSigma_dq, F))  # 3
            dF_dpa = -np.matmul(F, np.matmul(dSigma_dpa, F))  # 3
            ddetF_dq = detF * np.trace(np.matmul(Sigma, dF_dq))
            ddetF_dpa = detF * np.trace(np.matmul(Sigma, dF_dpa))
            # of Amplitude
            dA_dq = K / (2 * detF) * ddetF_dq  # 1
            dA_dpa = K / (2 * detF) * ddetF_dpa  # 1
            dA_dflux = K / flux # 1
            dA_dsersic = K / am * da_dn[i] # 1
            dA_drh = K / am * da_dr[i] # 1

            # Add derivatives to gaussians
            # As a list of just nonzero
            inds = [0, 3, 1] # slice into a flattened symmetric matrix to get unique components
            derivs = ([dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh] +
                      CW.flatten().tolist() +
                      dF_dq.flat[inds].tolist() +
                      dF_dpa.flat[inds].tolist())
            # as the 7 x 6 jacobian matrix
            # each row is a different theta and has
            # dA/dtheta, dx/dtheta, dy/dtheta, dFxx/dtheta, dFyy/dtheta, dFxy/dtheta
            jac = [[dA_dflux, 0, 0, 0, 0, 0],  # d/dFlux
                   [0, CW[0, 0], CW[1, 0], 0, 0, 0],  # d/dAlpha
                   [0, CW[0, 1], CW[1, 1], 0, 0, 0],  # d/dDelta
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


def compute_gig(gig, xpix, ypix, compute_deriv=True, **compute_keywords):
    """A slow loop of compute_gaussian() over all ImageGaussians in a
    GaussianImageGalaxy.
    """
    # Set up output
    assert len(xpix) == len(ypix)
    npix = len(xpix)
    image = np.zeros(npix)
    if compute_deriv:
        nskypar, nimpar = gig.gaussians.flat[0].derivs.shape
        gradients = np.zeros([nskypar, npix])
    else:
        gradients = None

    # Loop over on-image gaussians, accumulating image values and gradients
    for j, g in enumerate(gig.gaussians.flat):
        out = compute_gaussian(g, xpix, ypix, compute_deriv=compute_deriv,
                               **compute_keywords)
        if compute_deriv:
            I, dI_dphi = out
            gradients += np.matmul(g.derivs, dI_dphi)
        else:
            I = out

        image += I

    return image, gradients


def scale_matrix(q):
    #return np.array([[q**(-0.5), 0],
    #                [0, q**(0.5)]])
    return np.array([[1./q, 0],  #use q=(b/a)^0.5
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
