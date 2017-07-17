import numpy as np
import matplotlib.pyplot as pl
import proto


def set_params(source, theta):
    ra, dec, flux = theta
    source.ra = ra
    source.dec = dec
    source.flux = flux


def lnlike(thetas, sources, stamp):

    residual, partials = model_image(thetas, sources, stamp)
    chi = residual * stamp.ierr
    return -0.5 * np.sum(chi**2), np.sum(chi * partials, axis=0)


def model_image(thetas, sources, stamp, use_gradients=slice(None)):
    """Loop over all sources in a scene, subtracting each from the image and
    building up a gradient cube

    :returns residual:
        ndarray of shape (npix,)

    :returns partials:
        ndarray of shape (npix, nsource * ntheta)
    """
    ntheta = len(thetas[0])
    ngrad = len(sources) * ntheta
    dResid_dTheta = np.empty([stamp.npix, ngrad])

    for i, (theta, source) in enumerate(zip(thetas, sources)):
        set_params(source, theta)
        gig = proto.convert_to_gaussians(source, stamp)
        gig = proto.get_gaussian_gradients(source, stamp, gig)
        gig.ntheta = ntheta

        start = i * ntheta
        dResid_dtheta[:, start:start+ntheta] = (evaluate_gig(gig, stamp)[:, use_gradients])

    return stamp.residual, partials


def evaluate_gig(gig, stamp):
    """Evaluate one GaussianImageGalaxy, subtract it from the residual, and
    compute and return dResidual_dtheta
    """
    
    # R is the residual
    R = stamp.residual
    dR_dtheta = np.zeros([stamp.npix, gig.ntheta])

    for g in gig.gaussians.flat:
        I, dI_dphi = proto.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat)
        # Accumulate the derivatives w.r.t. theta from each gaussian
        # This matrix multiply can be optimized (many zeros in g.derivs)
        dR_dtheta += np.matmul(g.derivs.T, dI_dphi)
        R -= I

    # since R is stored in the stamp.residuals, we need only return the
    # derivatives for this gig
    return dR_dtheta


def work_plan(active_gigs, fixed_gigs, stamp):

    return chisq, dchisq_dtheta, residual 

        
if __name__ == "__main__":

    # get a source and a stamp
    source = proto.Galaxy()
    source.ngauss = 2
    source.radii = np.array([0,0.5])
    # source = proto.Star()
    stamp = proto.PostageStamp()

    # put an image in the stamp
    stamp.nx = 10
    stamp.ny = 10
    stamp.npix = int(stamp.nx * stamp.ny)
    stamp.xpix, stamp.ypix = np.meshgrid(np.arange(stamp.nx), np.arange(stamp.ny))
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])

    # Set source parameters
    theta = [10, 10, 1]
    set_params(source, theta)

    # center the image on the source
    stamp.crval = np.array([source.ra, source.dec])

    gig = proto.convert_to_gaussians(source, stamp)
    gig = proto.get_gaussian_gradients(source, stamp, gig)

    for g in gig.gaussians.flat:
        im, partial_phi = proto.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat)
        # This matrix multiply can be optimized (many zeros)
        partial_theta = np.matmul(g.derivs, partial_phi)

    fig, axes = pl.subplots(3, 3)
    for i, ddtheta in enumerate(partial_theta):
        ax = axes.flat[i]
        ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny))
        
    axes.flat[-1].imshow(im.reshape(stamp.nx, stamp.ny))
