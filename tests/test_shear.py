import sys
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

from forcepho import paths
from forcepho.data import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.sources import Scene, ConformalGalaxy, Galaxy

def test_shear():

    q, phi = 0.7, np.deg2rad(-20)
    im1, grad1, ngrad1, fig1 = dograd(q, phi, conformal=True)
    im2, grad2, ngrad2, fig2 = dograd(q, phi, conformal=False)

    assert np.allclose(im1, im2)
    assert np.allclose(grad1, ngrad1)
    assert np.allclose(grad2, ngrad2)


def numerical_image_gradients(theta0, delta, source=None, stamp=None):

    dI_dp = []
    for i, dp in enumerate(delta):
        theta = theta0.copy()
        theta[i] -= dp
        source.set_params(theta)
        imlo, _ = source.render(stamp, compute_deriv=False)
        theta[i] += 2 *dp
        source.set_params(theta)
        imhi, _ = source.render(stamp, compute_deriv=False)
        dI_dp.append((imhi - imlo) / (2 *dp))

    return np.array(dI_dp)


def dograd(q, phi, conformal=True, plot=False):
    stamp = PostageStamp(nx=50, ny=50)
    stamp.psf = PointSpreadFunction()
    stamp.scale = np.eye(2) * 50  #pixels per arcsec

    if conformal:
        source = ConformalGalaxy(splinedata=paths.galmixture+".h5")
        ep, ec = source.etas_from_qphi(q, phi)
        source.ep = ep
        source.ec = ec

    else:
        source = Galaxy(splinedata=paths.galmixture+".h5")
        source.q = q
        source.pa = phi
        
    source.ra = stamp.nx / 2.
    source.dec = stamp.ny / 2.
    source.flux = [1.0]
    source.sersic = 1.5
    source.rh = np.mean(source.rh_range)
    
    
    theta0 = source.get_param_vector().copy()
    delta = np.clip(theta0 * 1e-6, 1e-8, np.inf) 

    grad_num = numerical_image_gradients(theta0, delta, source, stamp)
    source.set_params(theta0)
    im, grad = source.render(stamp)

    if plot:
        fig = show(grad, grad_num, stamp)
    else:
        fig = None
    return im, grad, grad_num, fig


def show(grad, grad_num, stamp):
    npar = len(grad)
    fig, axes = pl.subplots(npar, 3, figsize=(6, 9),
                            sharex=True, sharey=True)
    for i in range(npar):
        g = grad[i,:].reshape(stamp.nx, stamp.ny)
        n = grad_num[i,:].reshape(stamp.nx, stamp.ny)
        c = axes[i, 0].imshow(n.T, origin='lower')
        fig.colorbar(c, ax=axes[i, 0])
        c = axes[i, 1].imshow(g.T, origin='lower')
        fig.colorbar(c, ax=axes[i, 1])
        c = axes[i, 2].imshow((n - g).T, origin='lower')
        fig.colorbar(c, ax=axes[i, 2])

    axes[0, 0].set_title('Numerical')
    axes[0, 1].set_title('Analytic')
    axes[0, 2].set_title('N - A')

    pl.show()
    return fig
    

def junk():

    #from forcepho.gaussmodel import convert_to_gaussians, get_gaussian_gradients, compute_gig
    #gig = convert_to_gaussians(source, stamp)
    #gig = get_gaussian_gradients(source, stamp, gig)
    # Do it!
    #image, grad = compute_gig(gig, stamp.xpix.flat, stamp.ypix.flat)

    #grad[3:5, :] = d/sqrt(q), d/dphi
    #grad[3,:] *= 0.5 / source.q
    # convert d/dq, d/dphi to d/deta_+, d/deta_x
    #q = (source.q)**2
    #phi = source.pa
    #sin2phi, cos2phi = np.sin(2 * phi), np.cos(2 * phi)
    #itlq = 1. / (2. * np.log(q))
    #ds_de = np.array([[-q * cos2phi, -q * sin2phi],
    #                  [sin2phi * itlq, -cos2phi * itlq ]])
    #grad[3:5, :] = np.matmul(ds_de.T, grad[3:5, :])
    #show(grad, grad_num, stamp)
    
    #return image, grad[self.use_gradients]

    pass
