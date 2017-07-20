import sys
import numpy as np
import matplotlib.pyplot as pl
import proto


class Scene(object):
    """A collection of sources describing the scene, and logic for parsing a
    parameter vector into individual source parameters
    """

    nsources = 4
    
    def thetas(self, params, stamps=None):
        pass


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
    return -0.5 * np.sum(chi**2), np.sum(chi * partials, axis=0)


def model_image(thetas, sources, stamp, use_gradients=slice(None)):
    """Loop over all sources in a scene, subtracting each from the image and
    building up a gradient cube.  Eventually everything interior to this should
    be moved to C++

    :returns residual:
        ndarray of shape (npix,)

    :returns partials:
        ndarray of shape (npix, nsource * ntheta)
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

    return chisq, dchisq_dtheta, residual 

        
if __name__ == "__main__":

    galaxy, new = True, True
    
    # get a stamp and put an image in it
    stamp = proto.PostageStamp()
    stamp.nx = 30
    stamp.ny = 30
    stamp.npix = int(stamp.nx * stamp.ny)
    stamp.xpix, stamp.ypix = np.meshgrid(np.arange(stamp.nx), np.arange(stamp.ny))
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])
    stamp.residual = np.zeros(stamp.npix)

    # get a source
    if galaxy:
        source = proto.Galaxy()
        source.ngauss = 4
        source.radii = np.arange(source.ngauss) * 0.5
        source.q = 0.5
        source.pa = np.deg2rad(30.)
        use = slice(0,5)
        theta = [1, 10., 10., 0.5, np.deg2rad(30.), 0., 0.]
    else:
        source = proto.Star()
        use = slice(0, 3)
        theta = [1, 10., 10., 0., 0., 0., 0.]

    # Set source parameters
    set_galaxy_params(source, theta)
    # center the image on the source
    stamp.crval = np.array([source.ra, source.dec])
    stamp.crval += 0.5 # move off-center by a half pix
    
    # New code
    if new:
        residual, partials = model_image([theta], [source], stamp,
                                        use_gradients=use)
        im = -residual
    
    
    #  --- OLD code ----
    if not new:
        gig = proto.convert_to_gaussians(source, stamp)
        gig = proto.get_gaussian_gradients(source, stamp, gig)

        for g in gig.gaussians.flat:
            im, partial_phi = proto.compute_gaussian(g, stamp.xpix.flat, stamp.ypix.flat)
            # This matrix multiply can be optimized (many zeros)
            partials = np.matmul(g.derivs, partial_phi)


    fig, axes = pl.subplots(3, 3)
    for i, ddtheta in enumerate(partials):
        ax = axes.flat[i]
        ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny))
    axes.flat[-1].imshow(im.reshape(stamp.nx, stamp.ny))    
