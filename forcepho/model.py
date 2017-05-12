try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad, jacobian
    _HAS_GRADIENTS = True
except(ImportError):
    import numpy as np
    _HAS_GRADIENTS = False


ln2pi = np.log(2 * np.pi)


class PixelResponse(object):


    def lnlike(self, params, source):
        image, imgrad = self.counts_and_gradients(params, source)
        delta = self.data - image
        chi = delta / self.unc
        lnlike = -0.5 * np.sum(chi**2)
        if imgrad is None > 1:
            lnlike_grad = None
        else:
            lnlike_grad = np.sum(chi / self.unc * imgrad , axis=0)

        return lnlike, lnlike_grad

    
    def counts_and_gradients(self, params, source):
        """Return the pixel response to the source object, as well as the
        gradients of the pixel response with respect to parameters of the
        source.
        """
        c = self.counts(params, source)
        if self.hasgrad and source.hasgrad:
            g = self.counts_gradient(params, source)
        else:
            g = None
        return c, g
    
    def counts_gradient(self, params, source):
        if self.hasgrad:
            return self._counts_gradient(params, source)
        else:
            return None

    @property
    def _counts_gradient(self):
        return jacobian(self.counts, argnum=0)


class GaussianMixtureResponse(PixelResponse):

    """An object which approximates the PSF by a mixture of gaussians.  This
    allows for analytic convolution with GaussianMixtureSources, under the
    assumption that the PSF does not change across the source.
    """

    hasgrad = _HAS_GRADIENTS

    def __init__(self, amplitudes=[], radii=[], mu=0., points=None):
        self.ncomp = len(amplitudes)
        self.means = np.array([[mu, mu] for i in range(self.ncomp)])
        self.covar = np.array([np.diag([r, r]) for r in radii])
        self.amplitudes = np.array(amplitudes)
    
    def convolve(self, params, source):
        """Convolve via sums of mean vectors and covariance matrices and products of amplitudes.
        """
        ns = source.ncomp
        nr = self.ncomp

        source_mu, source_sigma, source_amp = source.gaussians(params)
        mu = source_mu[None, :, :] + self.means[:, None, :]
        sigma = source_sigma[None, :, :, :] + self.covar[:, None, :, :]
        amplitude = source_amp[None, :] * self.amplitudes[:, None]
        return mu.reshape(nr*ns, 2), sigma.reshape(nr*ns, 2, 2), amplitude.reshape(nr*ns)

    def counts(self, params, source):
        x = self.points
        gaussians = self.convolve(params, source)
        mu, sigma, amplitude = gaussians
        #c = 1.0 * np.zeros(len(x))
        #for (m, s, a) in zip(*gaussians):
        #    c = c + a * normal(x - m, s)
        d = x[None, :, :] - mu[:, None, :]
        c = normal(d, sigma)
        return c.sum(axis=0)


#d = x[None, :, :] - mu[:, None, :]
#r = np.matmul(np.linalg.inv(sigma[:, None, :, :]), d[:, :, :, None])
#k = np.matmul(d[:, :, None, :], r)

def normal(x, sigma):
    """Calculate the normal density at x, assuming mean of zero.

    :param x:
        ndarray of shape (ngauss, npix, 2)

    :param sigma:
        ndarray pf shape (ngauss, 2, 2)

    returns density:
        ndarray of shape (ngauss, npix)
    """
    ln_density = -0.5 * np.matmul(x[:, :, None, :],
                                  np.matmul(np.linalg.inv(sigma[:, None, :, :]),
                                            x[:, :, :, None]))
    # sign, logdet = np.linalg.slogdet(sigma)
    # ln_density -= 0.5 * (logdet + ln2pi)
    # density = sign * np.exp(ln_density)
    density = np.exp(ln_density)[:, :, 0, 0]
    return density / np.sqrt(2 * np.pi * np.linalg.det(sigma)[:, None])


class PhonionPixelResponse(PixelResponse):

    """An object which applies the pixel response function to a set of point
    sources to compute the pixel counts (and gradients thereof with respect to
    the source properties).  This is incredibly general, since in principle the PRF can be
    different for every pixel.  It's also slow because to make an image one has
    to make a Python loop over PixelResponse objects.  
    """

    hasgrad = _HAS_GRADIENTS
    
    def __init__(self, mu, Sigma=[1., 1.]):
        """Initialize object with parameters of the pixel response function.
        Note that each mu and sigma corresponds to a single pixel.

        """
        self.mu = np.atleast_2d(mu)
        assert self.mu.shape[1] == 2

        s = np.atleast_1d(Sigma)
        assert s.shape[0] == 2
        if s.ndim == 1:
            self.Sigma = np.diag(s)
        elif ndim == 2:
            self.Sigma = s
        else:
            raise(ValueError, "Sigma must be one- or two-dimensional")

        #assert np.all((self.Sigma.shape) == 2)


    def counts(self, params, source):
        """Return the pixel response to the source with given params.

        Should allow here for vectorization (and use jacobian for the gradients)
        """
        rp = source.coordinates(params)  # (nphony, 2)
        weights = source.weights(params)  # (nphony)
        delta = rp[None, :, :] - self.mu[:, None, :]  # (npix, nphony, 2)
        # this is returns (npix, nphony, 1, 1)
        ln_density = -0.5 * np.matmul(delta[:, :, None, :],
                                  np.matmul(np.linalg.inv(self.Sigma[None, None, :, :]),
                                            delta[:, :, :, None]))
        #  and this returns (npix, nphoony)
        density = (weights[None, :] * np.exp(ln_density[:, :, 0, 0]) /
                   np.sqrt(2 * np.pi * np.linalg.det(self.Sigma)))
        return density.sum(axis=-1)
