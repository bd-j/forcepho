import autograd.numpy as np
from autograd import grad, elementwise_grad

class Source(object):

    n = 4.0
    x0 = 0.
    y0 = 0.
    theta = np.deg2rad(30)
    a = 2.
    b = 1.


    def __init__(self, nx=10, ny=10, **kwargs):
        self.points = self.draw_samples(nx, ny)

    def counts(self, params):
        rp = self.coordinates(params)
        c = np.sum(np.exp(-rp**2))# + rp[1, :]**2)
        return c

    @property
    def _counts_gradient(self):
        return grad(self.counts)
    
    def counts_gradient(self, params):
        return self._counts_gradient(params)

    def coordinates(self, params):
        rot = rotation_matrix(params[2])
        #scale = np.array([[params[0], 0],
        #                  [0, params[1]]])
        scale = scale_matrix(params[0], params[1])
    
        rp = np.dot(rot, np.dot(scale, self.points)) + params[-2:, None]
        return rp

    def draw_samples(self, nx, ny):
        x, y = np.meshgrid(np.linspace(-1, 1, nx),
                           np.linspace(-1, 1, ny),
                           sparse=False)
        #t = np.ones_like(x)
        #r = np.vstack([x, y, t]).reshape(3, -1)
        r = np.vstack([x, y]).reshape(2, -1)
        return r

    def sample_weights(self, x, y):
        z = np.hypot(x, y)** (1./self.n)
        w = np.exp(-z)
        return w

    def scale_matrix(self, a, b):
        return np.array([[a, 0],
                        [0, b]])

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])



class PixelResponse(object):

    def __init__(self, mu, Sigma=[1,1.]):
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

        
    def counts(self, params):
        rp = self.source.coordinates(params)
        c = np.sum(np.exp(-(rp - self.mu[:, None])**2))# + rp[1, :]**2)
        return c

    @property
    def _counts_gradient(self):
        return grad(self.counts)

    def counts_gradient(self, params):
        return self._counts_gradient(params)    

nx = ny = 10
x, y = np.meshgrid(np.linspace(-1, 1, nx),
                   np.linspace(-1, 1, ny),
                   sparse=False)
t = np.ones_like(x)
#r = np.vstack([x, y, t]).reshape(3, -1)
r = np.vstack([x, y]).reshape(2, -1)

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
    x0 = 0.5
    y0 = 0.
    theta = np.deg2rad(30)
    a = 2.
    b = 1.
    params = np.array([a, b, theta, x0, y0])
    
    lnp = countrate(params)
    counts_grad = grad(countrate)
    
    print(counts_grad(params))

    obj = Source()
    lnp = obj.counts(params)
    cg = grad(obj.counts)
    print(cg(params))
    print(obj.counts_gradient(params))

    pixel = PixelResponse(mu=[0., 0.])
    pixel.source = obj
    lnp = pixel.counts(params)
    print(pixel.counts_gradient(params))
