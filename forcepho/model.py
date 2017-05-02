try:
    import autograd.numpy as np
    from autograd import grad, elementwise_grad
    _HAS_GRADIENTS = True
except(ImportError):
    import numpy as np
    _HAS_GRADIENTS = False


ln2pi = np.log(2 * np.pi)
    
class GaussianMixturePSF(object)

    """An object which approximates the PSF by a mixture of gaussians.  This
    allows for analytic convolution with GaussianMixtureSources, under the
    assumption that the PSF does not change across the source.
    """

    def convolve(self, params):
        """Convolve via sums of mean vectors and covariance matrices and products of amplitudes.
        """
        ns = self.source.ncomp
        nr = self.ncomp

        source_mu, source_sigma, source_amp = self.source.gaussians(params)
        mu = source_mu[None, :, :] + self.means[:, None, :]
        sigma = source_sigma[None, :, :, :] + self.covar[:, None, :, :]
        amplitude = source_amp[None, :] * self.amplitudes[:, None]
        return mu.reshape(nr*ns, 2), sigma.reshape(nr*ns, 2, 2), amplitude.reshape(nr*ns)

    def counts(self, params):
        gaussians = self.convolve(params)
        mu, sigma, amplitude = gaussians
        c = np.zeros(len(x))
        for (m, s, a) in gaussians:
            c += a * normal(x - m, s)
        return c


def normal(x, sigma):
    """Calculate the normal density at x, assuming mean of zero.

    :param x:
        ndarray of shape (n, 2)
    """
    ln_density = -0.5 * np.matmul(x, np.matmul(np.linalg.inv(sigma), x.T))
    # sign, logdet = np.linalg.slogdet(sigma)
    # ln_density -= 0.5 * (logdet + ln2pi)
    # density = sign * np.exp(ln_density)
    density = np.exp(ln_density) / np.sqrt(2 * np.pi * np.linalg.det(sigma))
    return density


class PixelResponse(object):

    """An object which applies the pixel response function to a set of point
    sources to compute the pixel counts (and gradients thereof with respect to
    the source properties).  This is incredibly general, since the PRF can be
    different for every pixel
    """

    hasgrad = _HAS_GRADIENTS
    
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
        if source.hasgrad:
            g = self.counts_gradient(self.source.params)
        else:
            g = None
        return c, g

    def counts(self, params):
        """Return the pixel response to the source with given params.
        """
        rp = self.source.coordinates(params)
        weights = self.source.weights(params)
        c = np.sum(weights * np.exp(-np.sum((rp - self.mu[:, None])**2, axis=0)))
        return c

    @property
    def _counts_gradient(self):
        return grad(self.counts)

    def counts_gradient(self, params):
        if self.hasgrad:
            return self._counts_gradient(params)
        else:
            return None


class ImageModel(object):

    def __init__(self, pixel_list):
        self.pixels = pixel_list

    def counts(self, source):
        self.image = np.zeros(self.npix)
        for i, p in enumerate(self.pixels):  # So gross
            p.source = source
            self.image[i] = p.counts(source.params)

        return self.image

    def counts_and_gradients(self, source):
        if source.hasgrad:
            self.image = np.zeros([len(source.params) + 1, self.npix])
        else:
            self.image = np.zeros([1, self.npix])
        for i, p in enumerate(self.pixels):   # So gross
            v, g = p.counts_and_gradients(source)
            self.image[0, i] = v
            if source.hasgrad:
                self.image[1:,i] = g
        return self.image

    @property
    def npix(self):
        return len(self.pixels)    


class Likelihood(ImageModel):

    def __init__(self, pixel_list, Source, data, unc):
        self.pixels = pixel_list
        self.source = Source
        self.data = data
        self.unc = unc

    def lnlike(self, params):
        self.source.update_vec(params)
        image = self.counts_and_gradients(self.source)
        delta = self.data - image[0, :]
        chi = delta / self.unc
        lnlike = -0.5 * np.sum(chi**2)
        if image.shape[0] > 1:
            lnlike_grad = np.sum(chi / self.unc * image[1:, :] , axis=-1)
        else:
            lnlike_grad = None
        return lnlike, lnlike_grad

