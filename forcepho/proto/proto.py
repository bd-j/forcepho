# Some code along the lines of what we will do in C++

import numpy as np


def counts_pg_native(dx, dy, fxx, fyy, fxy, A, second_order=True):
    """Calculate the counts and gradient for one pixel, one gaussian.  This
    basically amounts to evalutating the gaussian (plus second order term) and
    the gradients with respect to the 6 input terms.

    :param dx, dy: 
        The location at which to evaluate the gaussian.  E.g.
        dx = x_{0,pixel} - x_{0,gaussian}

    :param fxx, fyy, fxy:
        Specification of the (inverse) covariance matrix for the gaussian

    :param A:
        The amplitude of the gaussian

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

    vx = fxx * dx + fxy * dy
    vy = fyy * dy + fxy * dx
    #G = np.exp(-0.5 * (dx*dx*fxx + 2*dx*dy*fxy + dy*dy*fyy))
    G = np.exp(-0.5 * (dx*vx + dy*vy))
 
    if second_order:
        H = 1 + (vx*vx + vy*vy -fxx -fyy) / 3.
    else:
        H = 1.0

    c_h = A * G
    C = c_h * H
    dC_dA = G * H
    dC_dx = C*vx - second_order * c_h * 2./3. * (fxx*vx +fxy*vy) 
    dC_dy = C*vy - second_order * c_h * 2./3. * (fyy*vy +fxy*vx)
    dC_dfx = -0.5*C*dx*dx - second_order * c_h * (1. + 2.*dx*vx) / 3.
    dC_dfy = -0.5*C*dy*dy - second_order * c_h * (1. + 2.*dy*vy) / 3.
    dC_dfxy = -1.0*C*dx*dy - second_order * c_h * (1. + 2.*dy*vy) / 3.

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


def counts_p(pixels, source):
    """Add all gaussians (and gradients thereof) for a given source to a given pixel

    :param pixels:
        Pixel centers, array of shape (2, Npix)

    :param source:
        Source parameters, array of shape (Ntheta,) with Ntheta usually 7
    """
    
    # These could be done with a `source` object with `as_mixture()` and
    # `mixture_jacobian()` methods
    gaussians = source_to_mixture(*source)  # shape (Ng, Nphi)
    jacobians = source_mixture_jacobian(*source) # shape (Ng, Nphi, Ntheta)

    # allocate output
    image = np.zeros(pixels.shape[-1])
    gradient = np.zeros(pixels.shape[-1], jacobians.shape[-1])

    for g, gaussian in enumerate(gaussians):
        # for p, mu in enumerate(pixels.T):
        c, grad = counts_pg(pixels, gaussian, jacobian=jacobians[g])
        image += c
        gradient += grad

    return image, gradient


def source_to_mixture(xs, ys, q, pa, n, rh, flux):
    """Calculate the parameters of a gaussian mixture from a given set of
    source parameters.  This should be a method of a `Source` class.
    """

    gaussians = np.empty([ng, nphi])

    return gaussians


def source_mixture_jacobian(xs, ys, q, pa, n, rh, flux):
    """Calculate the jacobian matrices of the transformation from source to
    gaussian parameters.  This should be a method of a `Source` class.  Also it
    should allow for the return of sparse matrices
    """

    jacobian = np.empty([ng, nphi, ntheta])


class Source(object):

    def __init__(self):
        pass

    def update(self, **params):
        for k, v in params.items():
            self.params[k] = v

    def as_mixture(self):
        gaussians = np.empty([ng, nphi])
        return gaussians
