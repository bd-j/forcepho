#------------
# script with utilities to numerically test gradients
# -----------

import numpy as np
import matplotlib.pyplot as pl
from forcepho import gaussmodel as gm
from demo_utils import Scene, make_stamp, negative_lnlike_stamp, negative_lnlike_nograd, make_image


def set_params(g, theta):
    g.amp, g.xcen, g.ycen, g.fxx, g.fyy, g.fxy = theta
    return g


def numerical_gaussian_gradients(g, xpix, ypix, theta0, delta):

    dI_dp = []
    for i, (p, dp) in enumerate(zip(theta0, delta)):
        theta = theta0.copy()
        #print(theta, dp, g.fxy)
        g = set_params(g, theta)
        F = np.array([[g.fxx, g.fxy], [g.fxy, g.fyy]])
        imlo, _ = gm.compute_gaussian(g, xpix, ypix)
        dlo = np.sqrt(np.linalg.det(F))
        theta[i] += dp
        g = set_params(g, theta)
        F = np.array([[g.fxx, g.fxy], [g.fxy, g.fyy]])
        imhi, _ = gm.compute_gaussian(g, xpix, ypix)
        dhi = np.sqrt(np.linalg.det(F))
        #print(theta, dp, g.fxy)
        print(i, imlo.sum() / imhi.sum(), dlo, dhi)
        dI_dp.append((imhi - imlo) / (dp))
        
    return np.array(dI_dp)


#compute_gaussian(g, xpix, ypix, second_order=True, compute_deriv=True):

def test_image_gaussian_gradients(dp=1e-5):
    nx, ny = 30., 30.
    ypix, xpix = np.meshgrid(np.arange(ny), np.arange(nx))
    
    galaxy = gm.Galaxy()
    galaxy.ngauss = 1
    galaxy.radii = np.arange(galaxy.ngauss) + 1.0
    galaxy.q = 0.5
    galaxy.pa = np.deg2rad(30.)
    galaxy.flux = 100
    
    # Get the transformation matrix
    D = np.eye(2)
    R = gm.rotation_matrix(galaxy.pa)
    S = gm.scale_matrix(galaxy.q)
    T = np.dot(D, np.dot(R, S))

    # get galaxy component means, covariances, and amplitudes in the pixel space
    gcovar = np.matmul(T, np.matmul(galaxy.covariances, T.T))
    gamps = galaxy.amplitudes
    gmean = np.zeros(2) + 15.

    
    gauss = gm.ImageGaussian()
    # Covariance matrix
    covar = gcovar[0] + np.eye(2)
    f = np.linalg.inv(covar)
    gauss.fxx = f[0, 0]
    gauss.fxy = f[1, 0]
    gauss.fyy = f[1, 1]
    print(covar)
    print(f)
    
    # Now get centers and amplitudes
    # TODO: Add gain/conversion from stamp to go from physical flux to counts.
    gauss.xcen, gauss.ycen = gmean
    am, al = 1., 1.
    gauss.amp = galaxy.flux * am * al * np.linalg.det(f)**(0.5) / (2 * np.pi)

    theta0 = np.array([gauss.amp, gauss.xcen, gauss.ycen, gauss.fxx, gauss.fyy, gauss.fxy])
    print(theta0)
    
    im, grad = gm.compute_gaussian(gauss, xpix, ypix)
    grad_num = numerical_gaussian_gradients(gauss, xpix, ypix, theta0, np.ones(6) * dp)

    
    fig, axes = pl.subplots(len(grad), 3, sharex=True, sharey=True)
    for i in range(len(grad)):
        c = axes[i, 0].imshow(grad_num[i,:,:].T, origin='lower')
        fig.colorbar(c, ax=axes[i, 0])
        c = axes[i, 1].imshow(grad[i, :, :].T, origin='lower')
        fig.colorbar(c, ax=axes[i, 1])
        c = axes[i, 2].imshow((grad_num[i,:,:] - grad[i, :, :]).T, origin='lower')
        fig.colorbar(c, ax=axes[i, 2])

    axes[0, 0].set_title('Numerical')
    axes[0, 1].set_title('Analytic')
    axes[0, 2].set_title('N - A')

    return fig, axes, grad, grad_num


def numerical_image_gradients(theta0, delta, scene=None, stamp=None):

    dI_dp = []
    for i, (p, dp) in enumerate(zip(theta0, delta)):
        theta = theta0.copy()
        imlo, _ = make_image(theta, scene, stamp)
        theta[i] += dp
        imhi, _ = make_image(theta, scene, stamp)
        dI_dp.append((imhi - imlo) / (dp))

    return np.array(dI_dp)


def test_image_gradients(ptrue, delta, scene=None, stamp=None):
    delta = np.ones_like(ptrue) * 1e-6
    #numerical
    grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
    image, grad = make_image(ptrue, scene, stamp)
    fig, axes = pl.subplots(len(ptrue), 3, sharex=True, sharey=True)
    for i in range(len(ptrue)):
        g = grad[i,:].reshape(stamp.nx, stamp.ny)
        c = axes[i, 0].imshow(grad_num[i,:,:].T, origin='lower')
        fig.colorbar(c, ax=axes[i, 0])
        c = axes[i, 1].imshow(g.T, origin='lower')
        fig.colorbar(c, ax=axes[i, 1])
        c = axes[i, 2].imshow((grad_num[i,:,:] - g).T, origin='lower')
        fig.colorbar(c, ax=axes[i, 2])

    axes[0, 0].set_title('Numerical')
    axes[0, 1].set_title('Analytic')
    axes[0, 2].set_title('N - A')



if __name__ == "__main__":

    fig, ax, grad, grad_num = test_image_gaussian_gradients(1e-7)
