try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad
except(ImportError):
    import numpy as np

from itertools import product
import time

import numpy as np    
import matplotlib.pyplot as pl

import source, model


if __name__ == "__main__":
    rh = 3 # half-light radius
    rho = 0.5 # minor/major axis ratio
    a = rh / np.sqrt(rho) # semi-major axis
    b = rh * np.sqrt(rho) # semi-minor axis
    
    theta = np.deg2rad(30)  # position angle (CCW from positive x-axis)
    x0 = 0.5  # center x
    y0 = -0.5  # center y
    ptrue = np.array([rho, theta, x0, y0, rh])


    # --- Set up the galaxy and pixels -----
    galaxy = source.PhonionSource(nx=50, ny=70, n=4.0)
    npx = npy = 40
    pixel_centers = np.array(list(product(np.arange(-npx/2, npx/2), np.arange(-npy/2, npy/2))))
    imod = [model.PhonionPixelResponse(px) for px in pixel_centers]
    

    import sys

    # ---- Fake image -----
    image = np.array([pix.counts(ptrue, galaxy) for pix in imod])
    unc = np.sqrt(image)
    coo_true = galaxy.coordinates(ptrue)

    for pix in imod:
        pix.data = pix.counts(ptrue, galaxy)
        pix.unc = np.sqrt(pix.data)


    # ---- Likelihood object and negative ln-likelihood for minimization
    def nll(params, obj):
        blob = list([pix.lnlike(params, obj) for pix in imod])
        v = np.array([b[0] for b in blob])
        g = np.array([b[1] for b in blob]))
        return -v, -g

    def nll_nograd(params, obj):
        blob = list([pix.lnlike(params, obj) for pix in imod])
        v = np.array([b[0] for b in blob])        
        return -v


    # --- Initial parameter value -----
    p0 = np.array([11.0, 5.1, theta/1.1, 0.3, -0.3])
    coo_start = galaxy.coordinates(p0)

    # ---- Optimization ------
    bounds = [(0, 100), (0., 100), (0, np.pi), (-10, 10), (-10, 10)]
    from scipy.optimize import minimize
    tstart = time.time()
    result = minimize(nll, p0, args=galaxy, jac=True, bounds=bounds,
                      options={'ftol': 1e-12})
    topt = time.time() - tstart
    pf = result.x
    coo_f = galaxy.coordinates(pf)
    print("done optimization with gradient in {}s".format(topt))

    galaxy.hasgrad = False
    tstart = time.time()
    result_nograd = minimize(nll_nograd, p0, jac=False, bounds=bounds,
                             options={'ftol': 1e-12})
    topt_nograd = time.time() - tstart
    print("done optimization without gradient in {}s".format(topt_nograd))
    galaxy.hasgrad = True

    sys.exit()

    # ---- Plot image gradients ----
    def plot_gradients(parvec, galaxy=galaxy, pixels=imod):

        blob = list([pix.lnlike(parvec, obj) for pix in pixels])
        v = np.array([b[0] for b in blob])
        g = np.array([b[1] for b in blob]))
        imgr = np.hstack([v[:, None], g]).T

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        parn = ['counts',
                '$\partial counts/ \partial rho$',#'$\partial counts / \partial b$',
                r'$\partial counts/ \partial \theta$',
                '$\partial counts / \partial x0$', '$\partial counts / \partial y0$']
        fig, axes = pl.subplots(2, 3, figsize=(20, 11))
        for i, ax in enumerate(axes.flat):
            if i == len(imgr):
                break
            c = ax.imshow(imgr[i, :].reshape(npx, npy).T, origin='lower')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="10%", pad=0.05)
            cbar = pl.colorbar(c, cax=cax)
            ax.set_title(parn[i])
        return fig

    #sys.exit()
    fig_true = plot_gradients(ptrue)
    fig_start = plot_gradients(p0)
    fig_fit = plot_gradients(pf)
    pl.show()

    # Everything below here  needs to be updated

    # -- Plot chisq gradients ------
    sys.exit()
    parn = [r'residual ($\Delta$)',
            r'$\partial \Delta/ \partial a$',r'$\partial \Delta / \partial b$',
            r'$\partial \Delta/ \partial \theta$',
            r'$\partial \Delta/ \partial x0$', r'$\partial \Delta / \partial y0$']
    fig, axes = pl.subplots(2, 3, figsize=(20, 11))
    for i, ax in enumerate(axes.flat):
        c = ax.imshow(imgr[i, :].reshape(npx, npy).T, origin='lower')
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.05)
        cbar = pl.colorbar(c, cax=cax)
        ax.set_title(parn[i])
