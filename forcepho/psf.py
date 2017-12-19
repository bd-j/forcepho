import numpy as np

__all__ = ["PoinstSpreadFunction", "make_psf"]


class PointSpreadFunction(object):
    """Gaussian Mixture approximation to a PSF.
    """

    def __init__(self, units='pixels'):
        self.ngauss = 1
        self.covariances = np.array(self.ngauss * [[[1.,0.], [0., 1.]]])
        self.means = np.zeros([self.ngauss, 2])
        self.amplitudes = np.ones(self.ngauss)

        self.units = units

def make_psf(answer, **kwargs):

    psf = PointSpreadFunction()
    x, y, vx, vy, vxy, amps = params_to_gauss(answer, **kwargs)
    # Build the covariance matrices
    cov = [np.array([[xv, xyv],[xyv, yv]]) for xv, yv, xyv in zip(vx, vy, vxy)]
    psf.ngauss = len(x)
    psf.means = np.array([x, y]).T
    psf.amplitudes = amps
    psf.covariances = np.array(cov)

    return psf


def params_to_gauss(answer, oversample=8, start=0, center=504):
    """Convert the fitted parameters to the parameters used in the PSF gaussian mixture.

    :returns mux:
        The center x of each gaussian, in detector coordinates (i.e. not PSF
        image pixels, but actual detector pixels)

    :returns muy:
        The center y of each gaussian, in detector coordinates (i.e. not PSF
        image pixels, but actual detector pixels)

    :returns vx:
        The 0,0 entry of the covariance matrix (sigma_x^2) in (detector pixels)^2

    :returns vy:
        The 1,1 entry of the covariance matrix (sigma_y^2) in (detector pixels)^2

    :returns vxy:
        The 1,0 or 0,1 entry of the covariance matrix (rho * sigma_x * sigma_y)
        in (detector pixels)^2

    :returns amp:
        The amplitude of the gaussians
    """
    params = answer['fitted_params'].copy()
    ngauss = len(params) / 6
    params = params.reshape(ngauss, 6)
    # is this right?
    #TODO: work out zero index vs 0.5 index issues
    # need to flip x and y here
    mu = (params[:,1:3][:, ::-1] + start - center) / oversample
    sy = params[:, 3] / oversample
    sx = params[:, 4] / oversample
    vxy = params[:, 5] * sx * sy
    amp = params[:, 0]

    return mu[:, 0], mu[:, 1], sx**2, sy**2, vxy, amp


def mvn_pdf(pos_x, pos_y, params):
    amp, mu_x, mu_y, sigma_x, sigma_y, rho = params
    A = amp / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1-rho**2))
    dx = (pos_x - mu_x) / sigma_x
    dy = (pos_y - mu_y) / sigma_y
    exparg = -0.5 / (1-rho**2) * (dx**2 + dy**2 - 2 * rho * dx * dy)
    return A * np.exp(exparg)


def mvn_pdf_2d(params, x_max, y_max):  # 0,1,...,x_max-1
    pos_x_range = np.arange(x_max)
    pos_y_range = np.arange(y_max)
    result = mvn_pdf(pos_x_range[:, None], pos_y_range[None, :], params)
    return result


def mvn_pdf_2d_mix_fn(num_mix, x_max, y_max):
    def ret_func(params):
        ans = np.zeros([x_max, y_max])
        for i in xrange(num_mix):
            ans += mvn_pdf_2d(params[(6*i):(6*i+6)], x_max, y_max)
        return ans
    return ret_func
