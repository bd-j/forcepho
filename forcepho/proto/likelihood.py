import numpy as np
import proto


__all__ = ["lnlike", "model_image", "evaluate_gig",
           "set_star_params", "set_galaxy_params"]


def set_star_params(source, theta):
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
    return -0.5 * np.sum(chi**2), np.sum(chi * partials, axis=-1)


def model_image(thetas, sources, stamp, use_gradients=slice(None)):
    """Loop over all sources in a scene, subtracting each from the image and
    building up a gradient cube.  Eventually everything interior to this should
    be moved to C++

    :returns residual:
        ndarray of shape (npix,)

    :returns partials:
        ndarray of shape (nsource * ntheta, npix)
    """
    ntheta = len(thetas[0][use_gradients])
    ngrad = len(sources) * ntheta
    gradients = np.empty([ngrad, stamp.npix])

    for i, (theta, source) in enumerate(zip(thetas, sources)):
        set_galaxy_params(source, theta)
        gig = proto.convert_to_gaussians(source, stamp)
        gig = proto.get_gaussian_gradients(source, stamp, gig)
        gig.ntheta = ntheta

        sel = slice(i * ntheta, (i+1) * ntheta)
        gradients[sel, :] = evaluate_gig(gig, stamp, use_gradients=use_gradients)

    return stamp.residual, gradients


def evaluate_gig(gig, stamp, use_gradients=slice(None)):
    """Evaluate one GaussianImageGalaxy, subtract it from the residual, and
    compute and return dResidual_dtheta
    """
    
    # R is the residual
    R = stamp.residual
    dR_dtheta = np.zeros([gig.ntheta, stamp.npix])

    for g in gig.gaussians.flat:
        I, dI_dphi = proto.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat)
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
