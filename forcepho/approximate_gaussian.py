from itertools import product

from scipy.special import factorial
import numpy as np
from functools import partial

import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import h5py

from scipy.special import gamma, gammainc, gammaincinv
from scipy.special import hyp1f1 as hyper
from scipy.optimize import minimize

x = np.linspace(1e-3, 8.0, int(1e3))
x = np.arange(0.0005, 10.0, 0.001)

# magic numbers
# minrad, maxrad, dlnr
# arpenalty, arscale
# asmooth
# xmax
# chsq weighting
# ar penalty form


def fit_gaussians(nsersic=4., rh=1., smoothing=0.0, radii=None,
                  ncomp=6, x=x, asmooth=0.0, arpenalty=0.0):
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
        #amp = np.linspace(np.log(0.01), np.log(maxrad), len(radii))
        amp = 1 -(radii - np.log(rh))**2
        amp[0] = amp[1]  # deal with zero radii
        amp = np.ones_like(radii) * 1e-5
        p0 = amp

    alpha = gammaincinv(2.0 * nsersic, 0.5)
    beta = gammaincinv(2.0 * nsersic, 0.90)
    rbeta = rh * (beta / alpha)**nsersic

        
    sersic = sersic_profile(x, n=nsersic, rh=rh, sigma=smoothing)
    extras = {'radii': radii, 'x': x, 'target': sersic,
              'smoothing':smoothing,
              'asmooth': asmooth, 'arpenalty': arpenalty, 'arscale': rbeta}
    partial_chisq = partial(chisq, **extras)

    result = minimize(partial_chisq, p0, method="powell")
    result = minimize(partial_chisq, result.x, method="BFGS", jac=False)   
    result = minimize(partial_chisq, result.x, method="CG", jac=False)   
    return result


def chisq(params, x=None, target=None, radii=None,
          smoothing=0.0, asmooth=0., arpenalty=0,
          arscale=1.0,
          return_models=False):
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
    delta = (gauss - target) / target
    weighted_chisq = np.sum(delta * delta * target * x) / np.sum(x * target) #+ 1e-3 * np.sum(amps**2)
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

    # extended wings penalty:
    if arpenalty > 0:
        #weighted_chisq += arpenalty * np.sum((amps * disps)**2)
        weighted_chisq += arpenalty * np.sum(amps * np.exp(disps/arscale))
        #weighted_chisq += arpenalty * np.sum(amps[disps > arscale])

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
        use = x < sigma * 15
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
                  ns=None, rh=None, chisq=None, amps=None, xmax=None):
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
        #print(r)
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
    if xmax is not None:
        rax.axvline(xmax, linestyle=":", color='r')
    rax.set_xscale('log')
    rax.set_xlabel('pixels')
    rax.set_ylim(-0.5, 0.5)
    [a.set_xlim(3e-2, x.max()) for a in [ax, rax]]


    alpha = gammaincinv(2 * ns, 0.5)
    xprime = (x / rh)**(1./ns) * alpha
    cfd_sersic = gammainc(2. * ns, xprime)
    #cfd_gauss = gauss_cfd(x, amps, radii)
    
    return fig


def gauss_cfd(x, amplitudes, radii):
    num = n.zeros_like(x)
    for a, r in zip(amplitudes, radii):
        num += a * gamma(1) * gammainc(1, 0.5 * (x/r)**2)
    return num / np.sum(amplitudes)


def plot_amps(radii, amps):
    rr = radii.copy()
    if radii[0] == 0.0:
        rr[0] = rr[1] / 2.0
    fig, axes = pl.subplots(2, 1)
    ax = axes[0]
    ax.plot(rr, amps)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('dispersion (pixels)')
    ax.set_ylabel('amplitude')
    ax = axes[1]
    ax.plot(rr, amps/rr)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('scale (pixels)')
    ax.set_ylabel('amplitude/scale')
   
    return fig



def fit_profiles(rgrid=np.arange(0.5, 4.25, 0.5),
                 ngrid = np.arange(1.0, 5.5, 1.0),
                 lnradii=np.arange(np.log(0.005), np.log(6.0), np.log(2.)),
                 smoothing=0.0, asmooth=0.0, arpenalty=0.0,
                 x=np.arange(0.0005, 10.0, 0.001)):

    # Decide whether to fit smoothed sersic profiles (smoothing > 0)
    #smoothing = 0.1 # in units of pixels
    outname = "mog_model.smooth={:2.1f}.h5".format(smoothing)
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
        alpha = gammaincinv(2.0 * ns, 0.5)
        beta = gammaincinv(2.0 * ns, 0.95)
        #xmax = rh * (beta / alpha)**ns
        #xmax = max(rh * 4, 2 * radii.max())
        xmax = 2 * radii.max()
        #xmax = 30.0
        print(ns, rh, xmax)
        xx = x[x < xmax]
        res = fit_gaussians(nsersic=ns, rh=rh, radii=lnradii,
                            x=xx, smoothing=smoothing, asmooth=asmooth,
                            arpenalty=arpenalty)
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

        fig = plot_profiles(x, sersic, gauss, sm_sersic=sm_sersic, xmax=xmax,
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

    # set the pixel array that will be used for calculating models
    
    # Set up gaussians in pixel space
    minrad, maxrad, dlnr = 0.01, 25.0, np.log(2)
    lnradii = np.arange(np.log(minrad), np.log(maxrad), dlnr)
    #lnradii = np.log(np.arange(minrad, maxrad, 0.5))
    x = np.arange(0.01, 50.0, 0.01)

    outname = fit_profiles(smoothing=0.25, lnradii=lnradii, x=x,
                           asmooth=1e-8, arpenalty=1e-6)

    #result = h5py.File(outname, "r")
    #profiles = PdfPages('mog_model_prof.pdf')
    #amps = PdfPages('mog_model_amps.pdf')
    #for i in len(result['nsersisc']):
        
