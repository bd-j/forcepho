from itertools import product

from scipy.special import factorial
import numpy as np
from functools import partial

import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import h5py

from scipy.special import gamma, gammaincinv
from scipy.special import hyp1f1 as hyper
from scipy.optimize import minimize

x = np.linspace(1e-3, 8.0, int(1e3))
x = np.arange(0.0005, 10.0, 0.001)


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

    sersic = sersic_profile(x, n=nsersic, rh=rh, sigma=smoothing)
    extras = {'radii': radii, 'x': x, 'target': sersic}
    partial_chisq = partial(chisq, **extras)
    bounds = npar * [(0, None)]

    result = minimize(partial_chisq, p0, method="powell")
    result = minimize(partial_chisq, result.x, method="BFGS", jac=False)   
    result = minimize(partial_chisq, result.x, method="CG", jac=False)   
    return result


def chisq(params, x=None, target=None, radii=None,
          return_models=False):
    """
    :param params:
        ln of the parameters describing the amplitudes and sigmas of the
        gaussian mixture
    """
    if radii is not None:
        disps = np.exp(radii)
        amps = np.exp(params[:len(radii)])
    else:
        amps = np.exp(params[:len(params)/2])
        disps = np.exp(params[len(params)/2:])
    # this is to account for two-d normalization and smoothing
    disps = np.hypot(disps, smoothing)
    amps /= disps * np.sqrt(2 * np.pi)
    gauss = normal_oned(x, 0.0, amps, disps)
    delta = gauss - target
    weighted_chisq = np.sum(delta * delta * x) / np.sum(x) #+ 1e-3 * np.sum(disps**2)
    weighted_chisq *= 1e-2
    if return_models:
        return weighted_chisq, x, target, gauss
    else:
        return  weighted_chisq 


def sersic_profile(x, n=4, rh=1, sigma=0.0, order=50):
    """Calculate a sersic profile, optionally with some small smoothing
    """
    alpha = gammaincinv(2 * n, 0.5)
    r0 = rh / alpha**n

    if sigma <= 0.0:
        return np.exp(-(x / r0)**(1.0 / n) + alpha)
    else:
        # xx = np.concatenate([x, np.atleast_1d(rh)])
        value = np.zeros_like(x)
        use = x < sigma * 20
        xx = x[use]
        #print(sigma, r0, rh, n)
        p = -0.5 * (xx / sigma)**2
        A = np.exp(p)
        total = np.zeros_like(xx)
        for k in range(order):
            k = k * 1.0
            mu = 1 + k/(2*n)
            term = (-1)**k / factorial(k)
            term *= (np.sqrt(2.) * sigma / r0)** (k / n)
            term *= gamma(mu)
            term *= hyper(mu, 1, -p)
            total += term
            #good = np.isfinite(term)
            #total[good] += term[good]
        value[use] = A * total * np.exp(alpha) # last term normalizes to unit intensity at the half-light radius
        # deal with large radii where the gaussian fails, by replacing with the unconvolved sersic
        bad = (~np.isfinite(value)) | (~use)
        value[bad] = np.exp(-(x[bad] / r0)**(1.0 / n) + alpha)
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

    # Decide whether to fit smoothed sersic profiles (smoothing > 0)
    smoothing = 0.1 # in units of pixels
    outname = "mog_model.smooth={:2.1f}.h5".format(smoothing)
    # set the pixel array that will be used for calculating models
    x = np.arange(0.0005, 10.0, 0.001)
    
    # Set up gaussians in pixel space
    minrad, maxrad, dlnr = 0.005, 6.0, np.log(2)
    lnradii = np.arange(np.log(minrad), np.log(maxrad), dlnr)
    if smoothing > 0:
        lnradii = np.insert(lnradii, 0, -np.inf)
    radii = np.exp(lnradii)

    # Grid in nsersic and rh
    rgrid = np.arange(0.5, 4.25, 0.5)
    ngrid = np.arange(1.0, 5.5, 1.0)

    nres = len(ngrid) * len(rgrid)
#    dt = np.dtype([('nsersic', np.float), ('rh', np.float), ('chisq', np.float),
#                   ('amplitudes', np.float, (len(radii),)),
#                   ('sersic', np.float, (len(x),)),
#                   ('model', np.float, (len(x),)),
#                   ('smooth_sersic', np.float, (len(x),))])
#    result = np.zeros(nres, dtype=dt)

    result = h5py.File(outname, 'w')
    result.create_dataset('radii', data=radii)
    result.create_dataset('x', data=x)
    result.attrs['smoothing'] = smoothing
    result.create_dataset('nsersic', shape=(nres,))
    result.create_dataset('rh', shape=(nres,))
    result.create_dataset('chisq', shape=(nres,))
    result.create_dataset('amplitudes', shape=(nres, len(radii)))
    result.create_dataset('truth', shape=(nres, len(x)))
    result.create_dataset('model', shape=(nres, len(x)))
    result.create_dataset('smooth_sersic', shape=(nres, len(x)))

    pdf = PdfPages('mog_model.pdf')
    # Loop over radii and sersic indices
    for i, (ns, rh) in enumerate(product(ngrid, rgrid)):
        print(ns, rh)
        xx = x[x < max(rh * 4, 1.1* radii.max())]
        res = fit_gaussians(nsersic=ns, rh=rh, radii=lnradii,
                            x=xx, smoothing=smoothing)
        result['nsersic'][i] = ns
        result['rh'][i] = rh
        result['amplitudes'][i] = np.exp(res.x)
        result['chisq'][i] = res.fun
        sersic = sersic_profile(x, n=ns, rh=rh, sigma=0)
        sm_sersic = sersic_profile(x, n=ns, rh=rh, sigma=smoothing)
        _, _, sm_sersic, gauss = chisq(res.x, radii=lnradii,
                                       x=x, target=sm_sersic,
                                       return_models=True)
        result['truth'][i] = sersic
        result['smooth_sersic'][i] = sm_sersic
        result['model'][i] = gauss
        result.flush()

        fig, ax = pl.subplots()
        ax.plot(x, sersic, label='Sersic:n={:3.1f}, rh={:3.2f}'.format(ns, rh))
        if smoothing > 0:
            ax.plot(x, sm_sersic, label='smooth Sersic')
        ax.plot(x, gauss, label='GaussMix')
        ax.axvline(rh * 4, linestyle=":", color='k')
        ax.set_xlabel('pixels')
        ax.set_title('chisq={}'.format(res.fun))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 1e5)
        ax.set_xlim(1e-3, x.max())
        ax.legend()
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()
    result.close()
