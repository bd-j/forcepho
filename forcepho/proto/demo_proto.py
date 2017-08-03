import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as pl
import proto

from likelihood import model_image, set_galaxy_params


def negative_lnlike_stamp(theta, source=None, stamp=None, free_inds=slice(0, 3)):
    stamp.residual = stamp.pixel_values.flatten()
    thetas = [np.append(np.array(theta), np.array([1., 0., 0., 0.]))]
    #thetas[0] *= 100
    residual, partials = model_image(thetas, [source], stamp)
    chi = residual * stamp.ierr
    return 0.5 * np.sum(chi**2), -np.sum(chi * stamp.ierr * partials[free_inds, :], axis=-1)


if __name__ == "__main__":

    galaxy, new = True, True
    
    # get a stamp and put an image in it
    stamp = proto.PostageStamp()
    stamp.nx = 30
    stamp.ny = 30
    stamp.npix = int(stamp.nx * stamp.ny)
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])
    stamp.residual = np.zeros(stamp.npix)
    stamp.psf = proto.PointSpreadFunction()

    # get a source
    if galaxy:
        source = proto.Galaxy()
        source.ngauss = 4
        source.radii = np.arange(source.ngauss) * 0.5
        source.q = 0.5
        source.pa = np.deg2rad(30.)
        use = slice(0,5)
        theta = [1, 10., 10., 0.5, np.deg2rad(30.), 0., 0.]
        label = ['$\psi$', '$x$', '$y$', '$q$', '$\\varphi$']
    else:
        source = proto.Star()
        use = slice(0, 3)
        theta = [100, 10., 10., 1., 0., 0., 0.]
        label = ['$\psi$', '$x$', '$y$']

    # Set source parameters
    set_galaxy_params(source, theta)
    # center the image on the source
    stamp.crval = np.array([source.ra, source.dec])
    stamp.crval += 0.5 # move off-center by a half pix
    
    # ----- New code ------
    if new:
        residual, partials = model_image([np.array(theta) * 1.25], [source], stamp,
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


    # --- Optimization -----
    stamp.pixel_values = -residual.reshape(stamp.nx, stamp.ny)
    err = stamp.pixel_values.max() * 1e-2
    #err = np.sqrt(stamp.pixel_values.flatten())
    stamp.ierr = np.ones(stamp.npix) / err

    if True:
        fig, axes = pl.subplots(3, 2)
        for i, ddtheta in enumerate(partials):
            ax = axes.flat[i+1]
            ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny).T, origin='lower')
            ax.text(0.1, 0.85, '$\partial I/\partial${}'.format(label[i]), transform=ax.transAxes)

        ax = axes.flat[0]
        ax.imshow(im.reshape(stamp.nx, stamp.ny).T, origin='lower')
        ax.text(0.1, 0.85, 'Model (I)'.format(label[i]), transform=ax.transAxes)
        
    nll = partial(negative_lnlike_stamp, source=source, stamp=stamp)
    p0 = np.array(theta[use])
    bounds = [(0, 1e4), (0., 30), (0, 30), (0, 1), (0, np.pi/2)]
    from scipy.optimize import minimize
    result = minimize(nll, p0 * 1.005, jac=True, bounds=bounds,
                      options={'ftol': 1e-22, 'gtol': 1e-22, 'disp':True,
                               'iprint': 1, 'maxcor': 20}
                      )
