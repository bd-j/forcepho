# -*- coding: utf-8 -*-

"""gaussmodel.py

Python implementation of the scene rendering, including convolution and
gradient calculation.  numba is used to speed up some of the loops.
"""

import numpy as np
#import numba
#from numba.experimental import jitclass

__all__ = ["ImageGaussian", "GaussianImageGalaxy",
           "convert_to_gaussians", "get_gaussian_gradients",
           "compute_gaussian"]


#image_gaussian_numba_spec = list(zip(['amp', 'xcen', 'ycen', 'fxx', 'fxy', 'fyy'],
#                                     [numba.float64, ]*6)) + [('derivs', numba.float64[:, :])]


#@jitclass(image_gaussian_numba_spec)
class ImageGaussian(object):
    """This is description of one Gaussian, in pixel locations, as well as the
    derivatives of this Gaussian wrt to the Scene model parameters.

    The gaussian is described by 6 parameters
    * Amplitude amp
    * Inverse Covariance matrix [[fxx, fxy], [fxy, fyy]]
    * center xcen, ycen

    In the Scene space there are 7 parameters, so the dgauss_dscene matrix is 6
    x 7 (mostly zeros)
    """

    def __init__(self):
        self.amp = 0.
        self.xcen = 0.
        self.ycen = 0.
        self.fxx = 0.
        self.fxy = 0.
        self.fyy = 0.

        # derivs = [dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh, D.flatten(),
        #           dF_dq.flat[inds], dF_dpa.flat[inds]]
        # This is the dGaussian_dScene Jacobian matrix, possibly sparse:
        #self.derivs = None
        #float dGaussian_dScene[NDERIV];


class GaussianImageGalaxy(object):
    """ A list of ImageGaussians corresponding to one galaxy, after image
    scalings, PSF, and in the pixel space.  Like `GaussianGalaxy` in the c++
    code.
    """
    id = 0

    def __init__(self, ngalaxy, npsf, id=None):
        self.id = id
        self.gaussians = []  # length [ngalaxy*npsf]


def convert_to_gaussians(source, stamp, compute_deriv=False):
    """Takes a set of source parameters into a set of ImagePlaneGaussians,
    including PSF, and keeping track of the dGaussian_dScene.

    Parameters
    ----------
    galaxy : An instance of sources.Source (or subclass)
        The source to convert to on-image gaussians.

    stamp : a stamps.PostageStamp() instance
        Must have a valid PointSpreadFunction and scale matrix.

    compute_deriv : bool, optional (default, False)
        Compute the Jacobian for the on-sky to on-image parameters for each
        ImageGaussian in the returned GaussianImageGalaxy

    Returns
    -------
    gig : An instance of GaussianImageGalaxy
        Effectively an array of ImageGaussians.
    """

    # get info for this stamp
    if type(stamp) is int:
        D = source.stamp_scales[stamp]
        psf = source.stamp_psfs[stamp][0]
        CW = source.stamp_cds[stamp]  # dpix/dra, dpix/ddec
        crpix = source.stamp_crpixs[stamp]
        crval = source.stamp_crvals[stamp]
        G = source.stamp_zps[stamp]
        stampid = stamp
        filter_index = source.stamp_filterindex[stamp]

    else:
        D = stamp.scale  # pix/arcsec
        psf = stamp.psf
        CW = stamp.dpix_dsky
        crpix = stamp.crpix
        crval = stamp.crval
        G = stamp.photocounts
        stampid = stamp.id
        filter_index = source.filter_index(stamp.filtername)

    # Get the transformation matrix
    R = rotation_matrix(source.pa)
    S = scale_matrix(source.q)
    T = fast_dot_dot_2x2(D, R, S)

    # get source component means, covariances, and amplitudes in pixel space
    scovar = fast_matmul_matmul_2x2(T, source.covariances, T.T)
    samps = source.amplitudes
    sky = np.array([source.ra, source.dec])
    smean = np.dot(CW, sky - crval) + crpix
    flux = np.atleast_1d(source.flux)
    if len(flux) == 1:
        flux = flux[0]
    else:
        flux = flux[filter_index]

    # Convert flux to counts
    flux *= G

    # get PSF component means and covariances in the pixel space
    if psf.units == 'arcsec':
        pcovar = fast_matmul_matmul_2x2(D, psf.covariances, D.T)
        pmeans = np.matmul(D, psf.means)
        # FIXME need to adjust amplitudes to still sum to one?
        pamps = psf.amplitudes
    elif psf.units == 'pixels':
        pcovar = psf.covariances
        pmeans = psf.means
        pamps = psf.amplitudes

    gig = GaussianImageGalaxy(source.n_gauss, psf.n_gauss,
                              id=(source.id, stampid))

    gig.gaussians = _convert_to_gaussians(source.n_gauss, psf.n_gauss,
                                          scovar, pcovar,
                                          smean, samps, pmeans, pamps,
                                          flux)

    if compute_deriv:
        gig = get_gaussian_gradients(source, stamp, gig)

    return gig


#@numba.njit
def _convert_to_gaussians(source_n_gauss, stamp_psf_n_gauss, scovar, pcovar,
                          smean, samps, pmeans, pamps, flux):
    """This is a helper function to `convert_to_gaussians` that wraps the
    nested loops in a form suitable for compilation by numba.
    """
    # bootstrap the list, so numba knows what type it will be
    gaussians_out = [ImageGaussian()]
    # Loop over source profile Gaussians, then psf Gaussians
    for i in range(source_n_gauss):
        # scovar = np.matmul(T, np.matmul(source.covariances[i], T.T))
        for j in range(stamp_psf_n_gauss):
            gauss = ImageGaussian()
            # is this needed or just debugging?
            # ifso need to add `id` field to jitclass
            #gauss.id = (source.id, stamp.id, i, j)
            # Convolve the jth Source component with the ith PSF component

            # Covariance matrix
            covar = scovar[i] + pcovar[j]
            f = fast_inv_2x2(covar)
            gauss.fxx = f[0, 0]
            gauss.fxy = f[1, 0]
            gauss.fyy = f[1, 1]

            # Now get centers and amplitudes
            gauss.xcen, gauss.ycen = smean + pmeans[j]
            am, al = samps[i], pamps[j]
            gauss.amp = flux * am * al * fast_det_2x2(f)**(0.5) / (2 * np.pi)

            # And add to list of gaussians
            gaussians_out += [gauss]

    return gaussians_out[1:]


def get_gaussian_gradients(source, stamp, gig):
    """Compute the Jacobian for dphi_i/dtheta_j where phi are the parameters of
    the Image Gaussian and theta are the parameters of the Source in the Scene.

    Parameters
    ----------
    source : An instance of sources.Source (or subclass)
        The source to convert to on-image gaussians.

    stamp : a stamps.PostageStamp() instance
        Must have a valid PointSpreadFunction and scale matrix.

    gig : An instance of GaussianImageGalaxy
        Effectively an array of ImageGaussians.  The computed Jacobians will be
        assigned to the `deriv` attribute of each `ImageGaussian` in the `gig`

    Returns
    -------
    gig : An instance of GaussianImageGalaxy
        The same as the input `gig`, but with the computed Jacobians assigned
        to the `deriv` attribute of each of the consitituent `ImageGaussian`s
    """
    # get info for this stamp
    if type(stamp) is int:
        D = source.stamp_scales[stamp]
        psf = source.stamp_psfs[stamp][0]
        CW = source.stamp_cds[stamp]  # dpix/dra, dpix/ddec
        #crpix = source.stamp_crpixs[stamp]
        #crval = source.stamp_crvals[stamp]
        G = source.stamp_zps[stamp]  #  physical to counts
        filter_index = source.stamp_filterindex[stamp]
    else:
        D = stamp.scale  # pix/arcsec
        psf = stamp.psf
        CW = stamp.dpix_dsky
        #crpix = stamp.crpix
        #crval = stamp.crval
        G = stamp.photocounts
        filter_index = source.filter_index(stamp.filtername)

    # Get the transformation matrix and other conversions
    R = rotation_matrix(source.pa)
    S = scale_matrix(source.q)
    T = fast_dot_dot_2x2(D, R, S)

    # And its derivatives with respect to scene parameters
    dS_dq = scale_matrix_deriv(source.q)
    dR_dpa = rotation_matrix_deriv(source.pa)
    dT_dq = fast_dot_dot_2x2(D, R, dS_dq)
    dT_dpa = fast_dot_dot_2x2(D, dR_dpa, S)

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
        flux = flux[filter_index]

    # get PSF component means and covariances in the pixel space
    if psf.units == 'arcsec':
        pcovar = fast_matmul_matmul_2x2(D, psf.covariances, D.T)
        # FIXME need to adjust amplitudes to still sum to one?
        pamps = psf.amplitudes
    elif psf.units == 'pixels':
        pcovar = psf.covariances
        pamps = psf.amplitudes

    derivs = _get_gaussian_gradients(source.n_gauss, psf.n_gauss,
                                     scovars, pcovar, samps, pamps,
                                     flux, G, T,
                                     dT_dq, dT_dpa, da_dn, da_dr, CW)

    # Unpack the results array returned by `_get_gaussian_gradients`
    for i in range(source.n_gauss):
        for j in range(psf.n_gauss):
            gig.gaussians[i * psf.n_gauss + j].derivs = derivs[i, j]

    return gig


#@numba.njit
def _get_gaussian_gradients(source_n_gauss, stamp_psf_n_gauss, scovars, pcovar,
                            samps, pamps, flux, G, T,
                            dT_dq, dT_dpa, da_dn, da_dr, CW):
    """Wrap the core parts of `get_gaussian_gradients` in form suitable for
    numba.
    """

    derivs = np.empty((source_n_gauss, stamp_psf_n_gauss, 7, 6))

    for i in range(source_n_gauss):
        scovar = scovars[i]
        scovar_im = fast_dot_dot_2x2(T, scovar, T.T)
        for j in range(stamp_psf_n_gauss):
            #gaussid = (source.id, stamp.id, i, j)

            # convolve the jth Source component with the ith PSF component
            Sigma = scovar_im + pcovar[j]
            F = fast_inv_2x2(Sigma)
            detF = fast_det_2x2(F)
            am, al = samps[i], pamps[j]
            K = flux * G * am * al * detF**(0.5) / (2 * np.pi)

            # Now get derivatives
            # Of F
            dSigma_dq = (fast_dot_dot_2x2(T, scovar, dT_dq.T) +
                         fast_dot_dot_2x2(dT_dq, scovar, T.T))
            dSigma_dpa = (fast_dot_dot_2x2(T, scovar, dT_dpa.T) +
                          fast_dot_dot_2x2(dT_dpa, scovar, T.T))
            dF_dq = -fast_dot_dot_2x2(F, dSigma_dq, F)  # 3
            dF_dpa = -fast_dot_dot_2x2(F, dSigma_dpa, F)  # 3
            ddetF_dq = detF * fast_trace_2x2(np.dot(Sigma, dF_dq))
            ddetF_dpa = detF * fast_trace_2x2(np.dot(Sigma, dF_dpa))
            # of Amplitude
            dA_dq = K / (2 * detF) * ddetF_dq  # 1
            dA_dpa = K / (2 * detF) * ddetF_dpa  # 1
            dA_dflux = K / flux  # 1
            dA_dsersic = K / am * da_dn[i]  # 1
            dA_drh = K / am * da_dr[i]  # 1

            # --- Add derivatives to gaussians ---
            # - As a list of just nonzero -
            # Need to slice into a flattened symmetric matrix to get unique
            # components
            #inds = [0, 3, 1]
            #derivs = ([dA_dflux, dA_dq, dA_dpa, dA_dsersic, dA_drh] +
            #          CW.flatten().tolist() +
            #          dF_dq.flat[inds].tolist() +
            #          dF_dpa.flat[inds].tolist())
            # - As the full 7 x 6 jacobian matrix -
            # Each row is a different theta and has
            # dA/dtheta, dx/dtheta, dy/dtheta, dFxx/dtheta, dFyy/dtheta, dFxy/dtheta
            jac = [[dA_dflux, 0, 0, 0, 0, 0],  # d/dFlux
                   [0, CW[0, 0], CW[1, 0], 0, 0, 0],  # d/dAlpha
                   [0, CW[0, 1], CW[1, 1], 0, 0, 0],  # d/dDelta
                   [dA_dq, 0, 0, dF_dq[0, 0], dF_dq[1, 1], dF_dq[1, 0]],  # d/dQ
                   [dA_dpa, 0, 0, dF_dpa[0, 0], dF_dpa[1, 1], dF_dpa[1, 0]],  # d/dPA
                   [dA_dsersic, 0, 0, 0, 0, 0],  # d/dSersic
                   [dA_drh, 0, 0, 0, 0, 0]]  # d/dRh
            derivs[i, j] = np.array(jac)
    return derivs


def compute_gaussian(*args, **kwargs):
    """Thin wrapper function that supplies all kwargs.
    Workaround for https://github.com/numba/numba/issues/3875
    """
    defaults = dict(second_order=True, compute_deriv=True,
                    use_det=False, oversample=False)
    defaults.update(kwargs)
    C, gradients = _compute_gaussian(*args, **defaults)

    if defaults['compute_deriv']:
        return C, gradients
    else:
        return C


#@numba.njit
def _compute_gaussian(g, xpix, ypix, second_order=True, compute_deriv=True,
                      use_det=False, oversample=False):
    """Calculate the counts and gradient for one pixel, one gaussian.  This
    basically amounts to evalutating the gaussian (plus second order term) and
    the gradients with respect to the 6 input terms.  Like `computeGaussian`
    and `computeGaussianDeriv` in the C++ code.

    Parameters
    ----------
    g : An instance of ImageGaussian()
        or an object that has the attributes `xcen`, `ycen`, `amp`, `fxx`,
        `fyy` and `fxy`.

    xpix : float or ndarray of shape (npix,)
        The x coordinate of the pixel(s) where counts and gradient are desired.
        Scalar or ndarray of shape npix.

    ypix : float or ndarray of shape (npix,)
        The y coordinate of the pixel(s) where counts and gradient are desired.
        Same shape as `xpix`.

    second_order: bool, optional (default: True)
        Whether to use the 2nd order correction to the integral of the gaussian
        within a pixel.

    compute_deriv : bool (optional, default: True)
        Whether to compute the derivatives of the counts with respect to the
        gaussian parameters.

    use_det : bool (otional, default: False)
        Whether to include the determinant of the covariance matrix when
        normalizing the counts and calculating derivatives.

    oversample : bool (optional, default: False)
        If this flag is set, the counts (and derivatives) will be calculated on
        a grid oversampled by a factor of two in each dimension (assuming the
        input xpix and ypix are integers), and then summed at the end to
        produce a more precise measure of the counts in a pixel.
        Not actually working yet.

    Returns
    -------
    counts : float or ndarray of shape (npix,)
        The counts in each pixel due to this gaussian, same shape as `xpix` and
        `ypix`.

    grad : ndarray of shape (nderiv, npix)
        The gradient of the counts with respect to the 6 parameters of the
        gaussian in image coordinates, ndarray of shape (nderiv, npix)
    """

    # All transformations should keep the determinant of the covariance matrix
    # the same, so that it can be folded into the amplitude, making the
    # following uninportant
    # inv_det = g.fxx*g.fyy + 2*g.fxy
    # norm = np.sqrt((inv_det / 2*np.pi))

    # Oversampling adds an extra dimension to xp; not Numba friendly
    # To enable this, one would also have to do the non-oversampled version with
    # the extra dimension
    #if oversample:
    #    xoff = np.array([0.25, -0.25, -0.25, 0.25])
    #    yoff = np.array([0.25, 0.25, -0.25, -0.25])
    #    xp = np.atleast_1d(xpix).reshape(-1,1) + xoff[..., :]
    #    yp = np.atleast_1d(ypix).reshape(-1,1) + yoff[..., :]
    #else:

    xp = xpix
    yp = ypix

    # --- Calculate useful variables ---
    dx = xp - g.xcen
    dy = yp - g.ycen
    vx = g.fxx * dx + g.fxy * dy
    vy = g.fyy * dy + g.fxy * dx
    Gp = np.exp(-0.5 * (dx*vx + dy*vy))
    # G = np.exp(-0.5 * (dx*dx*g.fxx + 2*dx*dy*g.fxy + dy*dy*g.fyy))
    H = np.ones_like(vx)
    root_det = 1.

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

    gradients = np.empty((6, len(C)))
    if compute_deriv:
        gradients[0][:] = dC_dA[:]
        gradients[1][:] = dC_dx[:]
        gradients[2][:] = dC_dy[:]
        gradients[3][:] = dC_dfx[:]
        gradients[4][:] = dC_dfy[:]
        gradients[5][:] = dC_dfxy[:]

    #if oversample:
    #    C = C.sum(axis=-1) / 4.0
    #    if compute_deriv:
    #       gradients = gradients.sum(axis=-1) / 4.0

    # numba needs a consistent number of return args
    return C, gradients


def compute_gig(gig, xpix, ypix, compute_deriv=True, **compute_keywords):
    """A slow loop of compute_gaussian() over all ImageGaussians in a
    GaussianImageGalaxy.

    Parameters
    ----------
    gig : An instance of GaussianImageGalaxy

    xpix : float or ndarray of shape (npix,)
        The x-coordinates of the pixels for fluxes are desired

    ypix : float or ndarray of shape (npix,)
        The y-coordinates of the pixels for fluxes are desired

    compute_deriv : bool, optional (default: True)
        Whether to compute flux gradients with respect to source parameters

    Extra Parameters
    ----------------
    Extra parameters are passed as keyword arguments to `compute_gaussian`

    Returns
    -------
    image : ndarray of shape (stamp.npix)
        The flux of this source in each pixel of the stamp

    gradients : ndarray of shape (nderiv, stamp.npix).  Optional.
        The gradients of the source flux in each pixel with respect to
        source parameters
    """
    # Set up output
    assert len(xpix) == len(ypix)
    npix = len(xpix)
    image = np.zeros(npix)
    if compute_deriv:
        nskypar, nimpar = gig.gaussians[0].derivs.shape
        gradients = np.zeros([nskypar, npix])
    else:
        gradients = None

    # Loop over on-image gaussians, accumulating image values and gradients
    for j, g in enumerate(gig.gaussians):
        out = compute_gaussian(g, xpix, ypix, compute_deriv=compute_deriv,
                               **compute_keywords)
        if compute_deriv:
            I, dI_dphi = out
            gradients += np.matmul(g.derivs, dI_dphi)
        else:
            I = out

        image += I

    return image, gradients


#@numba.njit
def scale_matrix(q):
    """q = sqrt(b/a)
    """
    return np.array([[1./q, 0],
                     [0., q]])

#@numba.njit
def rotation_matrix(theta):
    """theta: position angle in radians
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


#@numba.njit
def scale_matrix_deriv(q):
    """q = sqrt(b/a)
    """
    return np.array([[-1./q**2, 0], [0., 1]])


#@numba.njit
def rotation_matrix_deriv(theta):
    """theta: position angle in radians
    """
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]])


#@numba.njit
def fast_inv_2x2(A):
    """Fast inverse for a 2x2 matrix

    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_inv(A), np.linalg.inv(A))
    True
    """
    assert A.shape == (2, 2)
    inv = np.empty_like(A)

    inv[0, 0] = A[1, 1]
    inv[1, 0] = -A[1, 0]
    inv[0, 1] = -A[0, 1]
    inv[1, 1] = A[0, 0]
    inv /= A[0, 0]*A[1, 1] - A[1, 0]*A[0, 1]
    return inv


#@numba.njit
def fast_det_2x2(A):
    """Fast determinant for a 2x2 matrix.

    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_det_2x2(A), np.linalg.det(A))
    True
    """
    assert A.shape == (2, 2)
    return A[0, 0]*A[1, 1] - A[1, 0]*A[0, 1]


#@numba.njit
def fast_trace_2x2(A):
    """Fast trace for a 2x2 matrix.

    >>> A = np.array([[0.1,-0.3],[1.5,0.9]])
    >>> np.allclose(fast_trace_2x2(A), np.trace(A))
    True
    """
    assert A.shape == (2, 2)
    return A[0, 0] + A[1, 1]


#@numba.njit
def fast_dot_dot_2x2(a, b, c):
    """Fast dot of three 2x2 matrices.

    >>> A,B,C = np.random.rand(3,2,2)
    >>> np.allclose(fast_dot_dot_2x2(A,B,C), np.dot(A, np.dot(B.C)))
    True
    """
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert c.shape == (2, 2)

    return np.array([[(a[0,0]*b[0,0] + a[0,1]*b[1,0])*c[0,0] + (a[0,0]*b[0,1] + a[0,1]*b[1,1])*c[1,0],
                      (a[0,0]*b[0,0] + a[0,1]*b[1,0])*c[0,1] + (a[0,0]*b[0,1] + a[0,1]*b[1,1])*c[1,1]],
                    [(a[1,0]*b[0,0] + a[1,1]*b[1,0])*c[0,0] + (a[1,0]*b[0,1] + a[1,1]*b[1,1])*c[1,0],
                     (a[1,0]*b[0,0] + a[1,1]*b[1,0])*c[0,1] + (a[1,0]*b[0,1] + a[1,1]*b[1,1])*c[1,1]]])


#@numba.guvectorize([(numba.float64[:,:],numba.float64[:,:],numba.float64[:,:],numba.float64[:,:]),
#                    (numba.float32[:,:],numba.float32[:,:],numba.float32[:,:],numba.float32[:,:])],
#                   '(n,n),(n,n),(n,n)->(n,n)', nopython=True)
def fast_matmul_matmul_2x2(A, B, C):
    """Fast matmul(A, matmul(B,C)) for 2x2 matrices. Obeys np.matmul
    broadcasting semantics as long as the last two dimensions are shape (2,2).

    This is implemented as a generalized ufunc on fast_dot_dot_2x2, which
    automatically provides the broadcasting.  It does mean that we have to
    specify the type signatures manually, though.

    >>> A,B = np.random.rand(2,2,2)
    >>> C = np.random.rand(9,2,2)
    >>> np.allclose(fast_matmul_matmul_2x2(A,B,C), np.matmul(A, np.matmul(B,C)))
    True
    >>> np.allclose(fast_matmul_matmul_2x2(A,C,B), np.matmul(A, np.matmul(C,B)))
    True
    >>> np.allclose(fast_matmul_matmul_2x2(C,B,A), np.matmul(C, np.matmul(B,A)))
    True
    """
    #res[:] = fast_dot_dot_2x2(A, B, C)
    return np.matmul(A, np.matmul(B, C))