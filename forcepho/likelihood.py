import numpy as np
from . import gaussmodel as gm


__all__ = ["lnlike", "model_image", "evaluate_gig",
           "set_star_params", "set_galaxy_params"]


def set_star_params(source, theta):
    "Unused"
    flux, ra, dec = theta
    source.ra = ra
    source.dec = dec
    source.flux = flux


def set_galaxy_params(source, theta):
    flux, ra, dec, q, pa, sersic, rh = theta
    source.flux = flux
    source.ra = ra
    source.dec = dec
    source.q = q
    source.pa = pa
    source.sersic = sersic
    source.rh = rh


def lnlike(thetas, sources, stamps):
#def lnlike(pvec, scene, stamp):
    #sources, thetas = scene.sources, scene.thetas(pvec)
    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return -0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials, axis=-1)


def model_image(thetas, sources, stamp, use_gradients=slice(None), **extras):
    """Loop over all sources in a scene, subtracting each from the image and
    building up a gradient cube.  Eventually everything interior to this should
    be moved to C++, since the loop is very slow.

    :param thetas:
        The parameter vectors for the sources.  Iterable of length `nsource`.
        Each parameter vector must be of length 7.

    :param sources:
        The source objects.  Iterable of length `nsource`

    :param stamp:
        The PostageStamp containing the image data and from which sources will
        be subtracted to build up the residual.

    :param use_gradients:
        This is a set of indices (or slice object) that index the gradients you
        actually want to use.  All 7 gradients are always calculated for every
        source, but e.g. for stars you only care about 3 of them, or if some
        parameters are fixed you don't care about their gradients.

    :returns residual:
        ndarray of shape (npix,).  This is the result of *subtracting* all the
        gaussians of all the sources from the initial (input) value of
        stamp.residual.  Thus the initial value of stamp.residual should be the
        measured pixel values.  Note that stamp.residual will be modified.

    :returns partials:
        ndarray of shape (nsource * ntheta, npix)
    """
    ntheta = len(thetas[0][use_gradients])
    ngrad = len(sources) * ntheta
    gradients = np.empty([ngrad, stamp.npix])

    for i, (theta, source) in enumerate(zip(thetas, sources)):
        set_galaxy_params(source, theta)
        gig = gm.convert_to_gaussians(source, stamp)
        gig = gm.get_gaussian_gradients(source, stamp, gig)
        gig.ntheta = ntheta

        sel = slice(i * ntheta, (i+1) * ntheta)
        gradients[sel, :] = evaluate_gig(gig, stamp, use_gradients=use_gradients, **extras)

    return stamp.residual, gradients


def evaluate_gig(gig, stamp, use_gradients=slice(None), **extras):
    """Evaluate one GaussianImageGalaxy, subtract it from the residual, and
    compute and return dResidual_dtheta
    """

    # R is the residual
    R = stamp.residual
    dR_dtheta = np.zeros([gig.ntheta, stamp.npix])

    for g in gig.gaussians.flat:
        I, dI_dphi = gm.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat, **extras)
        # Accumulate the derivatives w.r.t. theta from each gaussian
        # This matrix multiply can be optimized (many zeros in g.derivs)
        dR_dtheta += np.matmul(g.derivs, dI_dphi)[use_gradients, :]
        R -= I

    # since R is stored in the stamp.residuals, we need only return the
    # derivatives for this gig
    return dR_dtheta


def work_plan(active_gigs, fixed_gigs, stamp):
    """Not actually written...
    """
    return chisq, dchisq_dtheta, residual
