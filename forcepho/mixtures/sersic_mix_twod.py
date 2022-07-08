# -*- coding: utf-8 -*-

import numpy as np

try:
    import jax.numpy as jnp
    from jax import random, lax
    import numpyro
    import numpyro.distributions as dist
    AUTOGRAD = True
except(ImportError):
    AUTOGRAD = False

__all__ = ["sersic_prediction", "sersic_model", "get_kernel"]


def sersic_prediction(xpix, ypix, x=0, y=0,
                      a=1, weight=1, lnW=None,
                      radii=[1],
                      g1=0, g2=0,
                      bg=0,
                      separate=False):
    """Predict the flux for a GMM model.  This method should match, in both math
    and parameter names, the specification of the model in `sersic_model`.  Note
    that vectors can be supplied for any of the parameters to implement a
    Gaussian mixture (where scalar parameters are repeated for as many Gaussians
    as necessary.)
    """
    # TODO: implement second order correction?

    if lnW is not None:
        weight = jnp.exp(lnW)
    radii = jnp.array(radii)
    norm = a * weight / (2 * np.pi * radii * radii)
    dx = (xpix[:, None] - x) / radii
    dy = (ypix[:, None] - y) / radii
    exparg = -0.5 * (dx**2 + dy**2)
    mu = norm * jnp.exp(exparg)
    if not separate:
        mu = jnp.sum(mu, axis=-1) + bg
    return mu


def sersic_model(data=None, xpix=None, ypix=None, unc=1,
                 ngauss_neg=0,
                 xcen=0, ycen=0, dcen=0,
                 radii=[1], gmax=0,
                 maxbg=0,
                 afix=1, amax=2,
                 prior="exp-squared", kernel_scale=3, lambda_smooth=1):
    """
    numpyro model for a 2d mixture of gaussians with optional constant
    background component. Assumes radii of the gaussians are fixed and circular,
    and that they are exactly centered.  Various priors are available

    Parameters
    ----------
    data : ndarray of shape (npix,)
        flux values

    xpix : ndarray of shape (npix,)
        x-coordinate of pixels

    ypix : ndarray of shape (npix,)
        y-coordinate of pixels

    ngauss : int
        number of Gaussian components

    afix : float or None
        If supplied, the *fixed* total combined amplitude of the gaussian
        mixture for dirichlet prior

    amax : float
        If the amplitude is not fixed, this gives the upper limit on the total
        combined amplitude of the mixture

    radii : sequence of floats
        The radii in pixels of each Gaussian

    prior : string
        The kind of joint prior to place on the amplitudes
    """
    # TODO: add a negative gaussian option?

    rg = jnp.array(radii)
    ngauss = len(rg)

    # Fit a background?
    if maxbg > 0:
        bg = numpyro.sample("bg", dist.Uniform(-maxbg, maxbg))
    else:
        bg = 0.0

    # Fix x,y?
    if dcen > 0:
        x = numpyro.sample("x", dist.Normal(xcen, dcen))
        y = numpyro.sample("y", dist.Normal(ycen, dcen))
    else:
        x = xcen
        y = ycen

    # Fix g?
    if gmax > 0:
        g1 = numpyro.sample("g1", dist.Uniform(-maxg, maxg))
        g2 = numpyro.sample("g2", dist.Uniform(-maxg, maxg))
    else:
        g1 = 0
        g2 = 0

    # Fix total amplitude?
    if afix:
        tot = afix
    else:
        tot = numpyro.sample("a", dist.Uniform(0.5, amax))

    # Component amplitudes
    if prior == "dirichlet":
        # Dirichlet for the weights; This is a true GMM where we
        # (weakly) identify the components using concentration
        concentration = jnp.linspace(2, 1, ngauss)
        w = numpyro.sample("weight", dist.Dirichlet(concentration))
    else:
        # How to add a smoothness prior for the weights?
        # perhaps sample from ratios, then do the cumulative thing.
        # OR, use a multivariate normal with covariance of nearby amplitudes.... a GP
        kernel = get_kernel(ngauss, prior, scale=kernel_scale)
        Sigma = lambda_smooth * jnp.array(kernel)
        lnW = numpyro.sample("lnW", dist.MultivariateNormal(loc=jnp.zeros(ngauss) - jnp.log(ngauss),
                                                            covariance_matrix=Sigma))
        w = jnp.exp(lnW)

    # Do the computation
    mu = sersic_prediction(xpix, ypix, x=x, y=y, a=tot, weight=w,
                           radii=rg, bg=bg)

    # --- add negative gaussians? ---
    if ngauss_neg > 0:
        raise NotImplementedError

    d = numpyro.sample("flux", dist.Normal(mu, unc), obs=jnp.array(data))


def get_kernel(ngauss, prior, scale=1):

    if prior == "triangular":
        # --- simple triangular kernel ---
        Sigma = (2 * np.diag(np.ones(ngauss)) +
                 scale * np.diag(np.ones(ngauss-1), -1) +
                 scale * np.diag(np.ones(ngauss-1), +1))

    elif prior == "finite-difference":
        D = (2 * np.diag(np.ones(ngauss)) -
             scale * np.diag(np.ones(ngauss-1), -1) -
             scale * np.diag(np.ones(ngauss-1), +1))
        Sigma = np.linalg.inv(np.dot(D.T, D))

    elif prior == "exp-squared":
        # --- RBF kernel ---
        # TODO: implement
        distance = np.fromfunction(lambda i, j: np.abs(i-j), (ngauss, ngauss))
        Sigma = np.exp(-distance**2/(2 * scale**2))

    s = np.diag(np.sqrt(np.diag(Sigma)))
    sinv = np.linalg.inv(s)
    Sigma = np.dot(sinv, np.dot(Sigma, sinv))

    return Sigma

def free_sersic_prediction(xpix, ypix, x=0, y=0,
                           a=1, weight=1, lnW=None,
                           radii=[1], scale=1.0,
                           g1=0, g2=0,
                           bg=0):
    """Predict the flux for a GMM model with non-circular shape/orientation.
    This method should match, in both math and parameter names, the
    specification of the model in `free_sersic_model`.  Note that vectors can be
    supplied for any of the parameters to implement a Gaussian mixture (where
    scalar parameters are repeated for as many Gaussians as necessary.)

    Parameters
    ----------
    xpix : ndarray of shape (npix,)
        The x coordinate of each pixel

    ypix : ndarray of shape (npix,)
        The y coordinate of each pixel

    x : float
        The x coordinate of the object and Gaussian centers

    y : float
        The y coordinate of the object and Gaussian centers

    a : float, optional (default, 1)
        An overall scaling for the amplitude of gaussian

    weight : ndarray of shape (ngauss,)
        The relative amplitude of each gaussian

    radii : sequence of length (ngauss,)
        The dispersion of each gaussian, in pixels.

    scale : float, optional (default: 1.0)
        An overall scaling for the gaussian dispersions

    g1 : float, optional (default: 0.0)
        The 1st component of the reduced shear vector
        (elongation along the PA=0 axis)

    g2 : float, optional (default: 0.0)
        The 2nd (imaginary) component of the reduced shear vector
        (elongation along the PA=45 deg axis)

    bg : float, optional (default: 0.0)
        Background pedestal value.
    """
    # TODO: implement second order correction?
    raise(NotImplementedError)

    if lnW is not None:
        weight = jnp.exp(lnW)
    invrsq = 1 / (scale * jnp.array(radii))**2

    Ksq = 1 / (1 - g1**2 - g2**2)
    #RS = jnp.array([[1+g1, g2], [g2, 1-g1]])
    #RSsq = Ksq * jnp.array([[(1 + g1)**2 + g2**2, 2*g2],
    #                        [2*g2, (1 - g1)**2 + g2**2]])
    #RSsqinv = Ksq * jnp.array([[(1 - g1)**2 + g2**2, -2*g2],
    #                              [-2*g2, (1 + g1)**2 + g2**2]])

    Fxx = invrsq * Ksq * ((1 - g1)**2 + g2**2)
    Fyy = invrsq * Ksq * ((1 + g1)**2 + g2**2)
    Fxy = invrsq * Ksq * (-2*g2)

    dx = (xpix[:, None] - x)
    dy = (ypix[:, None] - y)

    vx = Fxx * dx + Fxy * dy
    vy = Fyy * dy + Fxy * dx
    Gp = jnp.exp(-0.5 * (dx*vx + dy*vy))
    #root_det_F = jnp.sqrt(Fxx * Fyy - Fxy * Fxy)
    root_det_F = jnp.sqrt(invrsq)

    norm = a * weight * root_det_F / (2 * jnp.pi)
    mu = norm * Gp
    mu = jnp.sum(mu, axis=-1) + bg
    return mu
