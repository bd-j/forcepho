try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad
    _HAS_GRADIENTS = True
except:
    import numpy as np
    _HAS_GRADIENTS = False


class Source(object):

    # this is not the right wat to deal with parameters.
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


    def weights(self, x, y):
        """Optionally reweight the samples
        """
        return 1.0

    
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
