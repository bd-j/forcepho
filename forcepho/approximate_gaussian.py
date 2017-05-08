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
                  ncomp=6, x=x, asmooth=0.0):
    """Fit a mixture of gaussians to a Sersic profile.

    :param nsersic:
        The Sersic index of the profile to fit.

    :param rh:
        The half-light radius (in pixels) of the profile.

    :param smoothing:
        The gaussian smoothing (dispersion in pixels) to apply to both the
        sersic profile and the gaussians before fitting.

    :param radii: optional
        The dispersions of the gaussians to use in the MoG approximation.

    :param ncomp: optional
        The number of gaussians to use.  Overridden by `radii` if the latter is present.

    :param x:
        The vector of pixel locations at which the profiles will be compared.
    """
    maxrad = x.max()

    if radii is None:
        npar = ncomp * 2
        sigma = np.log(rh) + np.linspace(np.log(0.001), np.log(maxrad), ncomp)
        amp = np.linspace(np.log(0.01), np.log(maxrad), ncomp)
        p0 = np.concatenate([amp, sigma])
    else:
        npar = len(radii)
        amp = np.linspace(np.log(0.01), np.log(maxrad), len(radii))
        p0 = amp

    sersic = sersic_profile(x, n=nsersic, rh=rh, sigma=smoothing)
    extras = {'radii': radii, 'x': x, 'target': sersic,
              'smoothing':smoothing, 'asmooth': asmooth}
    partial_chisq = partial(chisq, **extras)
    bounds = npar * [(0, None)]

    result = minimize(partial_chisq, p0, method="powell")
    result = minimize(partial_chisq, result.x, method="BFGS", jac=False)   
    result = minimize(partial_chisq, result.x, method="CG", jac=False)   
    return result


def chisq(params, x=None, target=None, radii=None,
          smoothing=0.0, asmooth=0., return_models=False):
    """
    :param params:
        ln of the parameters describing the amplitudes and sigmas of the
        gaussian mixture.  Note that the amplitudes are the amplitudes for the
        equivalent 2-d gaussian, even though we're fitting in one-d
    """
    #print(asmooth)
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
    weighted_chisq = np.sum(delta * delta * x) / np.sum(x) #+ 1e-3 * np.sum(amps**2)
    weighted_chisq *= 1e-2

    # smoothness regularization
    if asmooth > 0:
        d = np.diag(np.zeros_like(amps) - 1, -1)[:-1, :-1]
        d += np.diag(np.zeros_like(amps) + 2, 0)
        d += np.diag(np.zeros_like(amps) - 1, 1)[:-1, :-1]
        a = np.log(amps)
        da = np.dot(d, a)
        pa = np.dot(a.T, np.dot(d.T, da))
        weighted_chisq += asmooth * pa

    if return_models:
        return weighted_chisq, x, target, gauss
    else:
        return  weighted_chisq 


def sersic_profile(x, n=4, rh=1, sigma=0.0, order=50):
    """Calculate a sersic profile, optionally with some small gaussian smoothing
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


def plot_profiles(x, sersic, gauss, sm_sersic=None, radii=None,
                  ns=None, rh=None, chisq=None):
    fig = pl.figure()
    ax = fig.add_axes((.1,.3,.8,.6))
    rax = fig.add_axes((.1,.1,.8,.2))

    ax.plot(x, sersic, label='Sersic:n={:3.1f}, rh={:3.2f}'.format(ns, rh))
    if  sm_sersic is not None:
        residual = gauss / sm_sersic - 1
        ax.plot(x, sm_sersic, label='smooth Sersic', linewidth=2)
    else:
        residual = gauss / sersic - 1
        
    ax.plot(x, gauss, label='GaussMix', alpha=0.7)
        
    for r in radii:
        ax.axvline(r, linestyle=":", color='k', alpha=0.1)
    ax.set_title('chisq={}'.format(chisq))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(2e-6, 9e4)
        
    ax.legend()
    ax.set_xticklabels([])
        
    rax.plot(x, residual)
    rax.set_ylabel('GM/smSersic - 1')
    rax.axvline(rh * 4, linestyle=":", color='k')
    rax.axvline(1.1 * radii.max(), linestyle=":", color='k')
    rax.set_xscale('log')
    rax.set_xlabel('pixels')
    rax.set_ylim(-0.25, 0.25)
    [a.set_xlim(3e-2, x.max()) for a in [ax, rax]]
    return fig

def plot_amps(radii, amps):
    fig, ax = pl.subplots()
    ax.plot(radii, amps)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('dispersion (pixels)')
    ax.set_label('amplitude')
    return fig



def fit_profiles(rgrid=np.arange(0.5, 4.25, 0.5),
                 ngrid = np.arange(1.0, 5.5, 1.0),
                 smoothing=0.0, asmooth=0.0):

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
    result.attrs['smooth_amplitudes'] = asmooth
    result.create_dataset('nsersic', shape=(nres,))
    result.create_dataset('rh', shape=(nres,))
    result.create_dataset('chisq', shape=(nres,))
    result.create_dataset('success', shape=(nres,))
    result.create_dataset('amplitudes', shape=(nres, len(radii)))
    result.create_dataset('truth', shape=(nres, len(x)))
    result.create_dataset('model', shape=(nres, len(x)))
    result.create_dataset('smooth_sersic', shape=(nres, len(x)))

    profiles = PdfPages('mog_model_prof.pdf')
    amps = PdfPages('mog_model_amps.pdf')
    # Loop over radii and sersic indices
    for i, (ns, rh) in enumerate(product(ngrid, rgrid)):
        print(ns, rh)
        xx = x[x < max(rh * 4, 1.1* radii.max())]
        res = fit_gaussians(nsersic=ns, rh=rh, radii=lnradii,
                            x=xx, smoothing=smoothing, asmooth=asmooth)
        result['nsersic'][i] = ns
        result['rh'][i] = rh
        result['amplitudes'][i] = np.exp(res.x)
        result['chisq'][i] = res.fun
        result['success'][i] = res.success
        sersic = sersic_profile(x, n=ns, rh=rh, sigma=0)
        sm_sersic = sersic_profile(x, n=ns, rh=rh, sigma=smoothing)
        cxns, _, sm_sersic, gauss = chisq(res.x, radii=lnradii, smoothing=smoothing,
                                          x=x, target=sm_sersic,
                                          return_models=True)
        result['truth'][i] = sersic
        result['smooth_sersic'][i] = sm_sersic
        result['model'][i] = gauss
        result.flush()

        fig = plot_profiles(x, sersic, gauss, sm_sersic=sm_sersic,
                            radii=radii, ns=ns, rh=rh, chisq=res.fun)
        profiles.savefig(fig)
        pl.close(fig)

        fig = plot_amps(radii, np.exp(res.x))
        fig.suptitle('ns={}, rh={}'.format(ns, rh))
        amps.savefig(fig)
        pl.close(fig)

    amps.close()
    profiles.close()
    result.close()
    return outname


if __name__ == "__main__":
    outname = fit_profiles(smoothing=0.1, asmooth=1e-8)

    #result = h5py.File(outname, "r")
    #profiles = PdfPages('mog_model_prof.pdf')
    #amps = PdfPages('mog_model_amps.pdf')
    #for i in len(result['nsersisc']):
        
