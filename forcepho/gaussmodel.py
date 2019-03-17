# Some code along the lines of what we will do in C++


import numpy as np
import numba

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


@numba.njit
def fast_inv_2x2(A):
    '''
    Fast inverse for a 2x2 matrix

    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_inv(A), np.linalg.inv(A))
    True
    '''
    assert A.shape == (2,2)
    inv = np.empty_like(A)
    
    inv[0,0] = A[1,1]
    inv[1,0] = -A[1,0]
    inv[0,1] = -A[0,1]
    inv[1,1] = A[0,0]
    inv /= A[0,0]*A[1,1] - A[1,0]*A[0,1]
    return inv


@numba.njit
def fast_det_2x2(A):
    '''
    Fast determinant for a 2x2 matrix.
    
    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_det_2x2(A), np.linalg.det(A))
    True
    '''
    assert A.shape == (2,2)
    return A[0,0]*A[1,1] - A[1,0]*A[0,1]


@numba.njit
def fast_trace_2x2(A):
    '''
    Fast trace for a 2x2 matrix.
    
    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_trace_2x2(A), np.trace(A))
    True
    '''
    assert A.shape == (2,2)
    return A[0,0] + A[1,1]


@numba.njit
def _convert_to_gaussians(gaussians, scovar, pcovar, smean, samps, pmeans, pamps, flux):
    '''
    This is a helper function to `convert_to_gaussians` that wraps the nested loops
    in a form suitable for compilation by numba.
    '''
    #gaussians = np.empty((source_ngauss,stamp_psf_ngauss), dtype=object)
    source_ngauss, stamp_psf_ngauss = gaussians.shape

    for i in range(source_ngauss):
        # scovar = np.matmul(T, np.matmul(source.covariances[i], T.T))
        for j in range(stamp_psf_ngauss):
            #gauss = ImageGaussian()
            gauss = gaussians[i,j]
            #gauss.id = (source.id, stamp.id, i, j)

            # Convolve the jth Source component with the ith PSF component

            # Covariance matrix
            covar = scovar[i] + pcovar[j]
            f = fast_inv_2x2(covar)
            gauss['f'][0] = f[0, 0]
            gauss['f'][1] = f[1, 0]
            gauss['f'][2] = f[1, 1]

            # Now get centers and amplitudes
            gauss['cen'][:] = smean + pmeans[j]
            am, al = samps[i], pamps[j]
            gauss['amp'] = flux * am * al * fast_det_2x2(f)**(0.5) / (2 * np.pi)

            # And add to list of gaussians
            #gaussians[i, j] = gauss

    return gaussians


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

    # Create an array that the numba function can use to conveniently output many variables
    gaussians_out = np.empty((source.ngauss,stamp.psf.ngauss),
                    dtype=[('amp',np.float),('cen',np.float,2),('f',np.float,3)])

    _convert_to_gaussians(gaussians_out, scovar, pcovar, smean, samps, pmeans, pamps, flux)

    gig = GaussianImageGalaxy(source.ngauss, stamp.psf.ngauss,
                              id=(source.id, stamp.id))
    for i in range(source.ngauss):
        for j in range(stamp.psf.ngauss):
            # Now unpack the array into the desired container object
            # An alternative would be to make ImageGaussian a jitclass, but those can be less stable
            go = gaussians_out[i,j]
            gauss = ImageGaussian()
            gauss.id = (source.id, stamp.id, i, j)
            gauss.amp = go['amp']
            gauss.fxx, gauss.fxy, gauss.fyy = go['f']
            gauss.xcen, gauss.ycen = go['cen']
            
            gig.gaussians[i,j] = gauss
    

    if compute_deriv:
        gig = get_gaussian_gradients(source, stamp, gig)

    return gig

@numba.njit
def _get_gaussian_gradients(source_ngauss, stamp_psf_ngauss, scovars, pcovar, T, samps, pamps, flux, G, dT_dq, dT_dpa, da_dn, da_dr, CW):
    '''
    Wrap the core parts of `get_gaussian_gradients` in a form suitable for numba.
    '''

    derivs = np.empty((source_ngauss, stamp_psf_ngauss, 7, 6))

    for i in range(source_ngauss):
        scovar = scovars[i]
        scovar_im = np.dot(T, np.dot(scovar, T.T))
        for j in range(stamp_psf_ngauss):
            #gaussid = (source.id, stamp.id, i, j)

            # convolve the jth Source component with the ith PSF component
            Sigma = scovar_im + pcovar[j]
            F = fast_inv_2x2(Sigma)
            detF = fast_det_2x2(F)
            am, al = samps[i], pamps[j]
            K = flux * G * am * al * detF**(0.5) / (2 * np.pi)

            # Now get derivatives
            # Of F
            dSigma_dq = (np.dot(T, np.dot(scovar, dT_dq.T)) +
                         np.dot(dT_dq, np.dot(scovar, T.T)))
            dSigma_dpa = (np.dot(T, np.dot(scovar, dT_dpa.T)) +
                          np.dot(dT_dpa, np.dot(scovar, T.T)))
            dF_dq = -np.dot(F, np.dot(dSigma_dq, F))  # 3
            dF_dpa = -np.dot(F, np.dot(dSigma_dpa, F))  # 3
            ddetF_dq = detF * fast_trace_2x2(np.dot(Sigma, dF_dq))
            ddetF_dpa = detF * fast_trace_2x2(np.dot(Sigma, dF_dpa))
            # of Amplitude
            dA_dq = K / (2 * detF) * ddetF_dq  # 1
            dA_dpa = K / (2 * detF) * ddetF_dpa  # 1
            dA_dflux = K / flux # 1
            dA_dsersic = K / am * da_dn[i] # 1
            dA_drh = K / am * da_dr[i] # 1

            # Add derivatives to gaussians
            # As a list of just nonzero
            inds = [0, 3, 1] # slice into a flattened symmetric matrix to get unique components
            #derivs = ([dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh] +
            #          CW.flatten().tolist() +
            #          dF_dq.flat[inds].tolist() +
            #          dF_dpa.flat[inds].tolist())
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
            derivs[i,j] = np.array(jac)
    return derivs


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

    derivs = _get_gaussian_gradients(source.ngauss, stamp.psf.ngauss, scovars, pcovar, T, samps, pamps, flux, G, dT_dq, dT_dpa, da_dn, da_dr, CW)

    # Unpack the results array returned by `_get_gaussian_gradients`
    for i in range(source.ngauss):
        for j in range(stamp.psf.ngauss):
            gig.gaussians[i,j].derivs = derivs[i,j]

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

@numba.njit
def scale_matrix(q):
    #return np.array([[q**(-0.5), 0],
    #                [0, q**(0.5)]])
    return np.array([[1./q, 0],  #use q=(b/a)^0.5
                    [0., q]])

@numba.njit
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

@numba.njit
def scale_matrix_deriv(q):
    #return np.array([[-0.5 * q**(-1.5), 0],
    #                [0, 0.5 * q**(-0.5)]])
    # use q=(b/a)**2
    return np.array([[-1./q**2, 0], [0., 1]])

@numba.njit
def rotation_matrix_deriv(theta):
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]])
