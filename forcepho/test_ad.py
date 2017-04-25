import autograd.numpy as np
from autograd import grad, elementwise_grad

from source import Source, scale_matrix, rotation_matrix
from model import PixelResponse, ImageModel, Likelihood

from source import sample_xy_grid, sample_sersic_flux
r = sample_xy_grid(10, 10)
r = sample_sersic_flux(10, 10, 4.0)

def countrate(params):

    #a, b, theta, mux, muy = params
    #rot = np.array([[np.cos(params[2]), -np.sin(params[2])],
    #                [np.sin(params[2]), np.cos(params[2])]])
    rot = rotation_matrix(params[2])
    #scale = np.array([[params[0], 0],
    #                  [0, params[1]]])
    scale = scale_matrix(params[0], params[1])
    
    rp = np.dot(rot, np.dot(scale, r)) + params[-2:, None]

    # convolution with gaussian centered at 0 and width 1.0 in each direction
    c = np.sum(np.exp(-rp**2))# + rp[1, :]**2)
    return c


if __name__ == "__main__":
    a = 10.
    b = 8.
    theta = np.deg2rad(30)
    x0 = 0.5
    y0 = -0.5
    ptrue = np.array([a, b, theta, x0, y0])

    # --- Testing junk -----
    #lnp = countrate(ptrue)
    #counts_grad = grad(countrate)
    #print(counts_grad(ptrue))
    #galaxy = Source(a=a, b=b, theta=theta, x0=x0, y0=y0)
    #pixel = PixelResponse(mu=[0., 0.])
    #pixel.source = galaxy
    #lnp = pixel.counts(ptrue)
    #print(pixel.counts_and_gradients(galaxy)[1])


    # --- Set up the galaxy and pixels -----
    galaxy = Source(a=a, b=b, theta=theta, x0=x0, y0=y0, nx=50, ny=70, n=1.0)
    npx = npy = 40
    pixel_list = [PixelResponse(mu=[i, j]) for i in range(-npx/2, npx/2) for j in range(-npy/2, npy/2)]
    imod = ImageModel(pixel_list)

    import sys
    #sys.exit()

    # ---- Fake image -----
    image = imod.counts(galaxy)
    unc = np.sqrt(image)
    coo_true = galaxy.coordinates(galaxy.params)

    # ---- Likelihood object and negative ln-likelihood for minimization
    model = Likelihood(pixel_list, galaxy, image, unc)
    def nll(params):
        v, g = model.lnlike(params)
        return -v, -g

    def nll_nograd(params):
        v, g = model.lnlike(params)
        return -v

    # --- Initial parameter value -----
    p0 = np.array([11.0, 5.1, theta/1.1, 0.3, -0.3])
    coo_start = galaxy.coordinates(p0)

    # ---- Optimization ------
    bounds = [(0, 100), (0., 100), (0, np.pi), (-10, 10), (-10, 10)]
    from scipy.optimize import minimize
    import time
    tstart = time.time()
    result = minimize(nll, p0, jac=True, bounds=bounds,
                      options={'ftol': 1e-12})
    topt = time.time() - tstart
    pf = result.x
    coo_f = galaxy.coordinates(pf)
    print("done optimization with gradient in {}s".format(topt))

    model.source.hasgrad = False
    tstart = time.time()
    result_nograd = minimize(nll_nograd, p0, jac=False, bounds=bounds,
                             options={'ftol': 1e-12})
    topt_nograd = time.time() - tstart
    print("done optimization without gradient in {}s".format(topt_nograd))
    model.source.hasgrad = True
    # Plot image gradients

    def plot_gradients(parvec, galaxy=galaxy, pixels=model):
        galaxy.update_vec(parvec)
        imgr = pixels.counts_and_gradients(galaxy)

        import matplotlib.pyplot as pl
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


    sys.exit()
    fig_true = plot_gradients(ptrue)
    fig_fit = plot_gradients(pf)
    
            
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
