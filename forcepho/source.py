try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad
    _HAS_GRADIENTS = True
except(ImportError):
    import numpy as np
    _HAS_GRADIENTS = False


from scipy.special import gammaincinv, gamma


class PhonionSource(object):

    """A Phonion source, basically a list of coordinates and associated
    weights.  The main outputs are through the ``coordinates`` method and the
    ``weights`` method.
    """
    # this is not the right way to deal with parameters.
    n = 4.0
    x0 = 0.
    y0 = 0.
    theta = np.deg2rad(30)
    a = 2.
    b = 1.

    hasgrad = _HAS_GRADIENTS

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

    def draw_samples(self, nx, ny):
        return sample_sersic_flux(nx, ny, self.n)
    
    def coordinates(self, params):
        """Return the detector coordinates of the source phonions given params
        """
        rot = rotation_matrix(params[2])
        #scale = np.array([[params[0], 0],
        #                  [0, params[1]]])
        scale = scale_matrix(params[0], params[1])
    
        rp = np.dot(rot, np.dot(scale, self.points)) + params[-2:, None]
        return rp


    def weights(self, params):
        """Optionally reweight the samples
        """
        return 1.0


class GaussianMixtureSource(object):

    """Mixture of arbitrary gaussians.  Mostly this returns a list of means,
    amplitudes, and covariance matrices through the ``covariance_matrices`` and
    ``amplitudes`` methods.
    """

    # this is not the right way to deal with parameters.
    ind_n = 5
    ind_rh = 4
    ind_rho = 0
    ind_pa = 1
    ind_mean = (2, 3)

    hasgrad = _HAS_GRADIENTS

    def __init__(self, amplitudes=[], radii=[], rh=1.0, n=4.0):
        self.rh = rh
        self.n = n
        self._radii = radii
        self._amplitudes = amplitudes
        self.ncomp = len(radii)
        self.covar = np.array([np.diag([r**2,r**2]) for r in radii])

    def amplitudes(self, params):
        return self._amplitudes

    def covariance_matrices(self, params):
        rot = rotation_matrix(params[self.ind_pa])
        sa = 1. / self.rh / np.sqrt(params[self.ind_rho])
        sb = np.sqrt(params[self.ind_rho]) / self.rh
        scale = scale_matrix(1. / np.sqrt(params[self.ind_rho]),
                             np.sqrt(params[self.ind_rho]))

        t = np.dot(rot, scale)
        covar = np.matmul(t, np.matmul(self.covar, t.T))
        return covar

    def means(self, params):
        return np.zeros([self.ncomp, 2]) + params[None, self.ind_mean]

    def gaussians(self, params):
        return self.means(params), self.covariance_matrices(params), self.amplitudes(params)


def scale_matrix(a, b):
        return np.array([[a, 0],
                        [0, b]])


def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])


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
