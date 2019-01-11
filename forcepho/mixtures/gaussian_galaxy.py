# --- Script for generating gaussian mixture approximations to Sersic indices ---


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

rgrid = np.arange(0.5, 5.25, 0.5)


def fit_gaussians(nsersic=4., rh=1., smoothing=0.0, radii=None,
                  ncomp=6, x=None, asmooth=0.0, arpenalty=0.0):
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

    :param asmooth:
        Strength of the smoothness regularization (based on penalizing large gradients in amplitudes)

    :param arpenalty:
        Strength of the penalty for large outer gaussians.
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
          smoothing=0.0,
          asmooth=0., arpenalty=0, arscale=1.0,
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
        sersic = np.exp(-(x / r0)**(1.0 / n) + alpha)
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
        bad = (~np.isfinite(value)) | (~use) | ((sersic > 1.5 * value) & (x > sigma))
        value[bad] = sersic[bad]
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
                  ns=0.0, rh=0.0, chisq=None, amps=None,
                  xmax=None, smoothing=None):
    fig = pl.figure()
    ax = fig.add_axes((.1,.5,.8,.4))
    cax = fig.add_axes((.1,.1,.8,.2))
    rax = fig.add_axes((.1,.3,.8,.2))
    
    ax.plot(x, sersic, label='Sersic:n={:3.1f}, rh={:3.3f}'.format(ns, rh))
    if  sm_sersic is not None:
        residual = gauss / sm_sersic - 1
        ax.plot(x, sm_sersic, label='smooth Sersic', linewidth=2)
    else:
        residual = gauss / sersic - 1
        
    ax.plot(x, gauss, label='GaussMix', alpha=0.7)
        
    for a, r in zip(amps, radii):
        #print(r)
        ax.axvline(r, linestyle=":", color='k', alpha=0.1)
        d = np.hypot(r, smoothing)
        ax.plot(x, normal_oned(x, 0.0, a/(d * np.sqrt(2. * np.pi)), d),
                               linestyle=":", color='k', alpha=0.1)
    ax.set_title('chisq={}'.format(chisq))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(2e-6, 9e4)
    ax.set_ylabel("I(r)/I(rh)")
        
    ax.legend()
    ax.set_xticklabels([])
        
    rax.plot(x, residual)
    rax.set_ylabel('GM/smSersic - 1')
    rax.axvline(rh * 4, linestyle=":", color='k')
    rax.axvline(1.1 * radii.max(), linestyle=":", color='k')
    if xmax is not None:
        rax.axvline(xmax, linestyle=":", color='r')
    rax.set_xscale('log')
    rax.set_ylim(-0.5, 0.5)
    rax.set_xticklabels([])

    f, cax = plot_cfd(x, ns, rh, amps, radii, smoothing, ax=cax)
    cax.set_xlabel('pixels')
    [a.set_xlim(x.min() * 0.8, x.max()*1.2) for a in [ax, rax, cax]]

    
    return fig


def rfrac_from_halfn(frac, n=4, rh=1):
    assert frac <= 1.0
    beta = gammaincinv(2 * n, frac)
    alpha = gammaincinv(2 * n, 0.5)
    r0 = rh / alpha**n
    r_frac = r0 * beta **n
    return r_frac


def plot_cfd(x, ns, rh, amps, radii, smoothing, ax=None):
    alpha = gammaincinv(2 * ns, 0.5)
    xprime = (x / rh)**(1./ns) * alpha
    cfd_sersic = gammainc(2. * ns, xprime)
    disps =  np.hypot(radii, smoothing)
    cfd_gauss = gauss_cfd(x, amps, radii)
    cfd_smgauss = gauss_cfd(x, amps, disps)

    if ax is None:
        fig, ax = pl.subplots()
    else:
        fig = None
    ax.plot(x, cfd_sersic, label='Sersic', linewidth=2, alpha=0.5)
    ax.plot(x, cfd_smgauss, label="smoothed gaussians", alpha=0.5)
    ax.plot(x, cfd_gauss, label="native gaussians", alpha=0.5)
    ax.axvline(smoothing, linestyle=':', color='blue', label='smoothing radius')
    ax.set_xscale('log')
    ax.set_xlim(x.min()*0.8, x.max()*1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('f(<r)/f(total)')
    ax.legend(fontsize='x-small', loc="upper left")
    return fig, ax
    

def gauss_cfd(x, amplitudes, radii):
    num = np.zeros_like(x)
    for a, r in zip(amplitudes, radii):
        if r <= 0.0:
            num += a
        else:
            num += a * gamma(1) * gammainc(1, 0.5 * (x/r)**2)
    return num / np.sum(amplitudes)


def amplitude_table(hname="gauss_gal_results/v3/mog_model.smooth=0.2.h5"):
    dat = h5py.File(hname, "r")
    radii = np.array(dat['radii'])
    rtext = ['A_r_{:3.1f}'.format(r) for r in radii]
    rr = ' '.join(['{:3.1f}'.format(r) for r in radii])
    header = ("# Amplitudes for gaussian mixture approximations to (smoothed) Sersic profiles.\n"
              "# The first two columns give the Sersic index and half light radius of the galaxy profile.\n"
              "# The subsequent columns give the relative weights of the 2-d symmetric gaussian with \n"
              "#   the given dispersions that best fits the galaxy profile.\n"
              "# The dispersions of each gaussian are fixed from profile to profile, and are\n"
              "# " + rr + "\n"
              "# Both the sersic profile and the gausians were convolved with a gaussian \n"
              "#   of dispersion {:3.2f} before fitting.\n"
              "# Produced by gaussian_galaxy.py\n\n").format(dat.attrs['smoothing'])
    cols = '# n  r_h  ' + ' '.join(rtext)
    fmt = ['{:3.1f}','{:4.2f}'] + len(radii) * ['{:5.3e}']
    fmt = '  '.join(fmt)
    with open(hname.replace('.h5', '.tbl'), 'w') as out:
        out.write(header + '\n')
        out.write(cols + '\n')
        for i, a in enumerate(dat['amplitudes']):
            vals = [dat['nsersic'][i], dat['rh'][i]] + (a/a.sum()).tolist()
            line = fmt.format(*vals)
            out.write(line + '\n')


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



def fit_profiles(outroot=None, rgrid=rgrid,
                 ngrid=np.arange(1.0, 5.5, 1.0),
                 lnradii=np.arange(np.log(0.005), np.log(6.0), np.log(2.)),
                 smoothing=0.0, asmooth=0.0, arpenalty=0.0,
                 x=np.arange(0.0005, 10.0, 0.001)):

    # Decide whether to fit smoothed sersic profiles (smoothing > 0)
    #smoothing = 0.1 # in units of pixels
    if outroot is None:
        outroot = "gauss_gal_results/sersic_mog_model.smooth={:2.4f}".format(smoothing)
    outname = outroot + '.h5'
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

    profiles = PdfPages(outroot+'.prof.pdf')
    amps = PdfPages(outroot+'.amps.pdf')
    cdf = PdfPages(outroot+'.cdf.pdf')
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
        # renormalize to unit total flux
        A = np.exp(res.x)
        result['amplitudes'][i] = A / A.sum()
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
                            radii=radii, ns=ns, rh=rh, chisq=res.fun, amps=np.exp(res.x),
                            smoothing=smoothing)
        profiles.savefig(fig)
        pl.close(fig)

        fig = plot_amps(radii, np.exp(res.x))
        fig.suptitle('ns={}, rh={}'.format(ns, rh))
        amps.savefig(fig)
        pl.close(fig)

        fig, ax = plot_cfd(x, ns, rh, np.exp(res.x), radii, smoothing)
        fig.suptitle('ns={}, rh={}'.format(ns, rh))
        cdf.savefig(fig)
        pl.close(fig)

    amps.close()
    profiles.close()
    cdf.close()
    result.close()
    return outname


if __name__ == "__main__":

    if False:
        # rh grid in pixels
        rgrid = np.arange(0.5, 10.25, 0.5)
        # sersic index grid
        ngrid = np.arange(1.0, 8.0, 0.5)

        # set the pixel scale array that will be used for calculating models
        x = np.arange(0.01, 64.0, 0.01)

        # Set up the dispersions of the zero-centered gaussians in pixel space
        minrad, maxrad, dlnr = 0.20, 28.0, np.log(2)
        lnradii = np.arange(np.log(minrad), np.log(maxrad), dlnr)

        # Sersic smoothing scale in pixels
        smoothing = 0.25

        # arcsec per pixel (0.032 for NIRCAM, 0.06 for XDF)
        pixscale = 0.03

        oname = "gauss_gal_results/sersic_mog.expanded.smooth={:2.4f}".format(smoothing * pixscale)
        outname = fit_profiles(outroot=oname, smoothing=smoothing * pixscale,
                               lnradii=lnradii + np.log(pixscale), ngrid=ngrid,
                               rgrid=rgrid*pixscale, x=x*pixscale,
                               asmooth=1e-8, arpenalty=1e-6)


    if True:
        rgrid = np.arange(0.5, 10.25, 1.0)
        ngrid = np.arange(5, 8.25, 0.5)

        maxrad = rfrac_from_halfn(0.9, rh=rgrid.max(), n=ngrid.max())
        x = np.arange(0.01, maxrad * 2, 0.01)
        minrad, maxrad, dlnr = 0.2, np.round(maxrad), np.log(3)
        lnradii = np.arange(np.log(minrad), np.log(maxrad) + dlnr, dlnr)
        lnradii = np.linspace(np.log(minrad), np.log(maxrad), 10)

        # Sersic smoothing scale in pixels
        smoothing = 0.25

        # arcsec per pixel (0.032 for NIRCAM, 0.06 for XDF)
        pixscale = 0.03

        oname = "test_mog"
        outname = fit_profiles(outroot=oname, smoothing=smoothing * pixscale,
                               lnradii=lnradii + np.log(pixscale), ngrid=ngrid,
                               rgrid=rgrid*pixscale, x=x*pixscale,
                               asmooth=1e-8, arpenalty=1e-6)

    #result = h5py.File(outname, "r")
    #profiles = PdfPages('mog_model_prof.pdf')
    #amps = PdfPages('mog_model_amps.pdf')
    #for i in len(result['nsersisc']):
