import autograd.numpy as np
from autograd import grad, elementwise_grad

from scipy.special import gammaincinv, gamma

class Source(object):

    # this is not the right wat to deal with parameters.
    n = 4.0
    x0 = 0.
    y0 = 0.
    theta = np.deg2rad(30)
    a = 2.
    b = 1.


    def __init__(self, nx=10, ny=10, **kwargs):
        self.points = self.draw_samples(nx, ny)
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v  in kwargs.items():
            self.__dict__[k] = v  # HAAAACK        

    def update_vec(self, vec):  # super hacky
        self.a = vec[0]
        self.b = vec[1]
        self.theta = vec[2]
        self.x0 = vec[3]
        self.y0 = vec[4]

        
    @property
    def params(self):
        return np.array([self.a, self.b, self.theta, self.x0, self.y0])

    def coordinates(self, params):
        """Return the detector coordinates of the source given params
        """
        rot = rotation_matrix(params[2])
        #scale = np.array([[params[0], 0],
        #                  [0, params[1]]])
        scale = scale_matrix(params[0], params[1])
    
        rp = np.dot(rot, np.dot(scale, self.points)) + params[-2:, None]
        return rp

    def draw_samples(self, nx, ny):
        return sample_sersic_flux(nx, ny, self.n)

    def weights(self, x, y):
        """Optionally reweight the samples
        """
        return 1.0

    def scale_matrix(self, a, b):
        return np.array([[a, 0],
                        [0, b]])

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])



class PixelResponse(object):
    
    def __init__(self, mu, Sigma=[1,1.]):
        """Initialize object with parameters of the pixel response function.
        """
        self.mu = np.array(mu)
        assert self.mu.shape == (2,)

        s = np.atleast_1d(Sigma)
        if s.ndim == 1:
            self.Sigma = np.diag(s)
        elif ndim == 2:
            self.Sigma = s
        else:
            raise(ValueError, "Sigma must be one or 2-d")

        #assert np.all((self.Sigma.shape) == 2)

    def counts_and_gradients(self, source):
        """Return the pixel response to the source object, as well as the
        gradients of the pixel response with respect to parameters of the
        source.
        """
        self.source = source
        c = self.counts(self.source.params)
        g = self.counts_gradient(self.source.params)
        return c, g

    def counts(self, params):
        """Return the pixel response to the source with given params.
        """
        rp = self.source.coordinates(params)
        # weights = self.source.weights(params)
        c = np.sum(np.exp(-np.sum((rp - self.mu[:, None])**2, axis=0)))
        return c

    @property
    def _counts_gradient(self):
        return grad(self.counts)

    def counts_gradient(self, params):
        return self._counts_gradient(params)    


class ImageModel(object):

    def __init__(self, pixel_list):
        self.pixels = pixel_list

    def counts(self, source):
        self.image = np.zeros(self.npix)
        for i, p in enumerate(self.pixels):
            p.source = source
            self.image[i] = p.counts(source.params)

        return self.image

    def counts_and_gradients(self, source):
        self.image = np.zeros([len(source.params) + 1, self.npix])
        for i, p in enumerate(self.pixels):
            v, g = p.counts_and_gradients(source)
            self.image[0, i] = v
            self.image[1:,i] = g
        return self.image

    @property
    def npix(self):
        return len(self.pixels)    
        
class Likelihood(object):

    def __init__(self, Pixels, Source, data, unc):
        self.model = Pixels
        self.source = Source
        self.data = data
        self.unc = unc

    def lnlike(self, params):
        self.source.update_vec(params)
        image = self.model.counts_and_gradients(self.source)
        lnlike = -0.5 * np.sum(((self.data - image[0, :]) / self.unc)**2)
        lnlike_grad = np.sum((self.data - image[1:, :]) / self.unc**2, axis=-1)
        return lnlike, lnlike_grad


def sample_sersic_flux(nr, nphi, nsersic):
    """Sample r uniformly in cumulative luminosity L(<R),
    sample phi uniformly in the interval (0, 2\,pi)

    :returns xy:
        The drawn r, phi values transformed to x,y coordinates.
    """
    clf = np.linspace(0.01, 0.95, nr)
    z = gammaincinv(2. * nsersic, clf)
    r = z**nsersic / gamma( 2 * nsersic)
    phi = np.linspace(0, 2 * np.pi, nphi)
    r, phi = np.meshgrid(r, phi, sparse=False)
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    xy = np.vstack([x, y]).reshape(2, -1)
    return xy


def sample_xy_grid(nx, ny):
    """ Sample over a girid from -1 to 1 in x and y
    """
    x, y = np.meshgrid(np.linspace(-1, 1, nx),
                       np.linspace(-1, 1, ny),
                       sparse=False)
    #t = np.ones_like(x)
    #r = np.vstack([x, y, t]).reshape(3, -1)
    r = np.vstack([x, y]).reshape(2, -1)
    return r

    
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


def transformation_matrix(params):
    a, b, theta, mux, muy = params
    return np.dot(translation_matrix(mux, muy),
                  np.dot(rotation_matrix(theta),
                         scale_matrix(a, b)))


def scale_matrix(a, b):
        return np.array([[a, 0],
                        [0, b]])


def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])



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

    galaxy = Source(a=a, b=b, theta=theta, x0=x0, y0=y0, nx=70, ny=50)
    nx = ny = 40
    pixel_list = [PixelResponse(mu=[i, j]) for i in range(-nx/2, nx/2) for j in range(-ny/2, ny/2)]
    pixels = ImageModel(pixel_list)


    image = pixels.counts(galaxy)
    unc = np.sqrt(image)

    model = Likelihood(pixels, galaxy, image, unc)

    def nll(params):
        v, g = model.lnlike(params)
        return -v, -g

    p0 = np.array([11.0, 5.1, 0.3, 0.3, -0.3])
    coo_true = galaxy.coordinates(ptrue)
    coo_start = galaxy.coordinates(p0)

    bounds = [(0, 100), (0., 100), (0, np.pi), (-10, 10), (-10, 10)]
    
    from scipy.optimize import minimize
    r = minimize(nll, p0, jac=True, bounds=bounds)

    galaxy.update_vec(ptrue)
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
