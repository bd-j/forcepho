import autograd.numpy as np
from autograd import grad, elementwise_grad

from scipy.special import gammaincinv, gamma

from source import Source, scale_matrix, roation_matrix
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
    
    lnp = countrate(ptrue)
    counts_grad = grad(countrate)
    
    print(counts_grad(ptrue))

    galaxy = Source(a=a, b=b, theta=theta, x0=x0, y0=y0)
    #lnp = galaxy.counts(params)
    #cg = grad(obj.counts)
    #print(cg(params))
    #print(galaxy.counts_gradient(params))

    pixel = PixelResponse(mu=[0., 0.])
    pixel.source = galaxy
    lnp = pixel.counts(ptrue)
    print(pixel.counts_and_gradients(galaxy)[1])


    # --- Set up the galaxy and pixels -----
    galaxy = Source(a=a, b=b, theta=theta, x0=x0, y0=y0, nx=50, ny=70, n=1.0)
    nx = ny = 40
    pixel_list = [PixelResponse(mu=[i, j]) for i in range(-nx/2, nx/2) for j in range(-ny/2, ny/2)]
    pixels = ImageModel(pixel_list)

    import sys
    #sys.exit()

    # ---- Fake image -----
    image = pixels.counts(galaxy)
    unc = np.sqrt(image)
    coo_true = galaxy.coordinates(galaxy.params)

    # ---- Likelihood object and negative ln-likelihood for minimization
    model = Likelihood(pixels, galaxy, image, unc)
    def nll(params):
        v, g = model.lnlike(params)
        return -v, -g

    # --- Initial parameter value -----
    p0 = np.array([11.0, 5.1, 0.3, 0.3, -0.3])
    coo_start = galaxy.coordinates(p0)

    # ---- Optimization ------
    bounds = [(0, 100), (0., 100), (0, np.pi), (-10, 10), (-10, 10)]
    from scipy.optimize import minimize
    result = minimize(nll, p0, jac=True, bounds=bounds)
    pf = result.x
    coo_f = galaxy.coordinates(pf)

    # Plot image gradients

    def plot_gradients(parvec, galaxy, pixels):
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
            c = ax.imshow(imgr[i, :].reshape(nx, ny).T, origin='lower')
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="10%", pad=0.05)
            cbar = pl.colorbar(c, cax=cax)
            ax.set_title(parn[i])
        return fig


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
        c = ax.imshow(imgr[i, :].reshape(nx, ny).T, origin='lower')
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.05)
        cbar = pl.colorbar(c, cax=cax)
        ax.set_title(parn[i])
