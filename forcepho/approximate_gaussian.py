from itertools import product

from scipy.special import factorial
import numpy as np
from functools import partial


from scipy.special import gamma, gammaincinv
from scipy.special import hyp1f1 as hyper

from scipy.optimize import minimize

x = np.linspace(1e-3, 8.0, int(1e3))
x = np.arange(0.0005, 8.0, 0.001)

def fit_gaussians(nsersic=4., rh=1., smoothing=0.0, radii=None,
                  ncomp=6, x=x):

    maxrad = x.max()

    if radii is None:
        npar = ncomp * 2
        sigma = np.log(rh) + np.linspace(np.log(0.01), np.log(maxrad), ncomp)
        amp = np.linspace(np.log(0.01), np.log(maxrad), ncomp)
        p0 = np.concatenate([amp, sigma])
    else:
        npar = len(radii)
        amp = np.linspace(np.log(0.01), np.log(maxrad), len(radii))
        p0 = amp

    extras = {'nsersic': nsersic, 'rh': rh, 'smoothing': smoothing,
              'radii': radii, 'x': x}
    partial_chisq = partial(chisq, **extras)
    bounds = npar * [(0, None)]

    result = minimize(partial_chisq, p0, method="powell")
    result = minimize(partial_chisq, result.x, method="BFGS", jac=False)   
    result = minimize(partial_chisq, result.x, method="CG", jac=False)   
    return result


def chisq(params, nsersic=4, rh=1, smoothing=0.0, radii=None,
          x=np.linspace(1e-3, 8.0, int(1e3)), return_models=False):
    """
    :param params:
        ln of the parameters describing the amplitudes and sigmas of the
        gaussian mixture
    """

    sersic = sersic_profile(x, n=nsersic, rh=rh, sigma=smoothing)
    if radii is not None:
        disps = np.exp(radii)
        amps = np.exp(params[:len(radii)])
    else:
        amps = np.exp(params[:len(params)/2])
        disps = np.exp(params[len(params)/2:])
    amps /= disps * np.sqrt(2 * np.pi)
    disps = np.hypot(disps, smoothing)
    gauss = normal_oned(x, 0.0, amps, disps)
    delta = gauss - sersic
    weighted_chisq = np.sum(delta * delta * x) / np.sum(x) #+ 1e-3 * np.sum(disps**2)
    weighted_chisq *= 1e-2
    if return_models:
        return weighted_chisq, x, sersic, gauss
    else:
        return  weighted_chisq 


def sersic_profile(x, n=4, rh=1, sigma=0.0, order=100):
    """Calculate a sersic profile, optionally with some small smoothing
    """
    alpha = gammaincinv(2 * n, 0.5)
    r0 = rh / alpha**n

    if sigma <= 0.0:
        return np.exp(-(x / r0)**(1.0 / n) + alpha)
    else:
        # xx = np.concatenate([x, np.atleast_1d(rh)])
        xx = x
        print(sigma, r0, rh, n)
        p = -0.5 * (xx / sigma)**2
        A = np.exp(p)
        total = 0
        for k in range(order):
            k = k * 1.0
            mu = 1 + k/(2*n)
            term = (-1)**k / factorial(k)
            term *= (np.sqrt(2.) * sigma / r0)** (k / n)
            term *= gamma(mu)
            term *= hyper(mu, 1, -p)
            total += term
        value = A * total * np.exp(alpha)
        return value #[:-1] / value[-1], value[-1]


def normal_oned(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)



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


if __name__ == "__main__":

    # Set up gaussians in pixel space
    minrad, maxrad, dlnr = 0.001, 4.0, np.log(2)
    lnradii = np.arange(np.log(minrad), np.log(maxrad), dlnr)
    radii = np.exp(lnradii)
    
    # Grid in nsersic and rh
    rgrid = np.arange(0.25, 4.25, 0.25)
    ngrid = np.arange(1.0, 5.0, 0.5)

    nres = len(ngrid) * len(rgrid)
    dt = np.dtype([('nsersic', np.float), ('rh', np.float), ('chisq', np.float),
                   ('amplitudes', np.float, (len(radii),)),
                   ('truth', np.float, (len(x),)),
                   ('model', np.float, (len(x),))])
                   
    result = np.zeros(nres, dtype=dt)
    
    # 
    for i, (ns, rh) in enumerate(product(ngrid, rgrid)):
        print(ns, rh)
        x = np.arange(0.0005, rh * 4, 0.001)
        res = fit_gaussians(nsersic=ns, rh=rh, radii=lnradii, x=x)
        result[i]['nsersic'] = ns
        result[i]['rh'] = rh
        result[i]['amplitudes'] = res.x
        result[i]['chisq'] = res.fun
        chi, xx, sersic, gauss = chisq(res.x, nsersic=ns, rh=rh, radii=lnradii, x=x,
                                       return_models=True)
        result[i]['truth'] = sersic
        result[i]['model'] = gauss
