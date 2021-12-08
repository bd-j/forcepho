#------------
# script with utilities to numerically test gradients
# -----------

import numpy as np
import matplotlib.pyplot as pl
from forcepho import gaussmodel as gm
from forcepho.sources import Galaxy, Scene
from forcepho.stamp import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.likelihood import lnlike_multi, make_image, WorkPlan


splinedata = "test_data/sersic_mog_model.smooth=0.0150.h5"

def set_params(g, theta):
    g.amp, g.xcen, g.ycen, g.fxx, g.fyy, g.fxy = theta
    return g


def numerical_image_gradients(theta0, delta, scene=None, stamp=None):

    dI_dp = []
    for i, dp in enumerate(delta):
        theta = theta0.copy()
        theta[i] -= dp
        imlo, _ = make_image(scene, stamp, Theta=theta)
        theta[i] += 2 *dp
        imhi, _ = make_image(scene, stamp, Theta=theta)
        dI_dp.append((imhi - imlo) / (2 *dp))

    return np.array(dI_dp)


def test_image_gradients(ptrue, delta=None, scene=None, stamp=None):
    delta = np.ones_like(ptrue) * 1e-6
    #numerical
    grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
    image, grad = make_image(scene, stamp, Theta=ptrue)
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


def get_scene(ngal, nx, bands=["A"], **extras):
    sources = []
    xc = np.linspace(0, nx, ngal + 2)[1:-1]
    for x in xc:
        s = Galaxy(splinedata=splinedata, filters=bands)
        s.ra = nx / 2
        s.dec = x
        s.sersic=2
        s.rh = 0.10
        s.q = 0.9
        s.pa = 0.5
        s.flux = len(bands) * [1]
        sources.append(s)

    scene = Scene(sources)
    return scene
    

def get_stamps(nstamp, nx, npsf, bands=["A"], scene=None, pixscale=0.05, **extras):
    stamps = []
    for i in range(nstamp):
        s = PostageStamp()
        s.filtername = np.random.choice(bands)
        s.scale = 1.0/pixscale * np.eye(2)
        s.nx = s.ny = nx
        s.ypix, s.xpix = np.meshgrid(np.arange(s.ny), np.arange(s.nx))
        s.pixel_values = np.random.uniform(0, 1, size=s.shape)
        s.ierr = np.sqrt(s.pixel_values).flatten()
        s.psf = get_psf(npsf)
        s.photocounts = i + 1
        if scene is not None:
            im, _ = make_image(scene, s)
            s.pixel_values = im
        
        stamps.append(s)
    return stamps


def get_psf(npsf):
    psf = PointSpreadFunction()
    psf.ngauss = npsf
    psf.means = np.zeros([npsf, 2])
    psf.covariances = (np.arange(npsf) + 1)[:, None, None] * np.eye(2)
    psf.amplitudes = np.ones(npsf) / npsf
    return psf



if __name__ == "__main__":
    ngal = 1
    nx = 50
    nstamp = 2
    npsf = 1
    bands = ["flux"]

    scene = get_scene(ngal, nx, bands=bands)
    stamps = get_stamps(nstamp, nx, npsf, bands=bands, scene=scene)
    
    ptrue = scene.get_all_source_params().copy()
    test_image_gradients(ptrue, delta=None, scene=scene, stamp=stamps[1])
    pl.show()
    #fig, ax, grad, grad_num = test_image_gaussian_gradients(1e-7)
