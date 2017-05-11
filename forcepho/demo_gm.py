try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad
except(ImportError):
    import numpy as np

import matplotlib.pyplot as pl
import time

from source import PhonionSource, GaussianMixtureSource
from model import PixelResponse, ImageModel, Likelihood


def plot_gradients(imgr, npx=40, npy=40):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    parn = ['counts',
            '$\partial counts/ \partial a$','$\partial counts / \partial b$',
            r'$\partial counts/ \partial \theta$',
            '$\partial counts / \partial x0$', '$\partial counts / \partial y0$']
    fig, axes = pl.subplots(2, 3, figsize=(20, 11))
    for i, ax in enumerate(axes.flat):
        c = ax.imshow(imgr[i, :].reshape(npx, npy).T, origin='lower')
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.05)
        cbar = pl.colorbar(c, cax=cax)
        ax.set_title(parn[i])
    return fig


if __name__ == "__main__":
    a = 10. # semi-major axis
    b = 8.  # semi-minor axis
    theta = np.deg2rad(30)  # position angle (CCW from positive x-axis)
    x0 = 0.5  # center x
    y0 = -0.5  # center y
    ptrue = np.array([a, b, theta, x0, y0])


    # --- Set up the galaxy and pixels -----
    minrad, maxrad, dlnr = 0.20, 28.0, np.log(2)
    lnradii = np.arange(np.log(minrad), np.log(maxrad), dlnr)
    lnradii = np.insert(lnradii, 0, -np.inf)
    radii = np.exp(lnradii)


    psf = model.GaussianMixtureResponse(amplitudes=[1.0, 0.5], radii=[0.1, 0.5])
    npx = npy = 40
    points = np.array(list(product(np.arange(-npx/2, npx/2), np.arange(-npy/2, npy/2))))
    psf.points = points

    galaxy = source.GaussianMixtureSource(radii=radii, amplitudes=amplitudes)#, x0=x0, y0=y0, nx=50, ny=70, n=1.0)
    psf.source = galaxy


    c = psf.counts(ptrue)
    cj = psf.counts_gradients(ptrue)
     imgr = np.vstack([c, cj.T])

    fig = plot_gradients(imgr)
