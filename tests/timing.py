import sys, os, time
from copy import deepcopy
import numpy as np
from os.path import join as pjoin
import inspect
import timeit

from forcepho import paths
from forcepho.data import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.sources import Galaxy, Scene
from forcepho.likelihood import WorkPlan, lnlike_multi, make_image, plan_sources
from forcepho.gaussmodel import convert_to_gaussians, get_gaussian_gradients, compute_gaussian


# time image generation and likelihood call as function of
# - pix per stamp
# - N_stamps
# - N_galaxies
# - gaussians per psf
# - gaussians per galaxy
#
# time convert_to_gaussians and get_gaussian_gradients as fn of
# - gaussians per psf
# - gaussians per galaxy
#
# time compute_gaussian as a fn of
# - pix per stamp

bands = "abcdefghijkl"
splinedata = paths.galmixtures[1]
#splinedata = 'sersic_mog_model.smooth=0.0150.h5'

# --- Utilities ---

def get_scene(ngal, nx, nband=1, **extras):
    sources = []
    xc = np.linspace(0, nx, ngal + 2)[1:-1]
    for x in xc:
        s = Galaxy(splinedata=splinedata, filters=list(bands)[:nband])
        s.ra = nx / 2
        s.dec = x
        s.sersic=2
        s.rh = 0.10
        s.q = 0.9
        s.pa = 0.5
        s.flux = nband * [1]
        sources.append(s)

    scene = Scene(sources)
    return scene
    

def get_stamps(nstamp, nx, npsf, nband=1, pixscale=0.05, **extras):
    stamps = []
    for i in range(nstamp):
        s = PostageStamp()
        s.filtername = bands[np.random.choice(nband)]
        s.scale = 1.0/pixscale * np.eye(2)
        s.nx = s.ny = nx
        s.ypix, s.xpix = np.meshgrid(np.arange(s.ny), np.arange(s.nx))
        s.pixel_values = np.random.uniform(0, 1, size=s.shape)
        s.ierr = np.sqrt(s.pixel_values).flatten()
        s.psf = get_psf(npsf)
        stamps.append(s)
    return stamps


def get_psf(npsf):
    psf = PointSpreadFunction()
    psf.ngauss = npsf
    psf.means = np.zeros([npsf, 2])
    psf.covariances = (np.arange(npsf) + 1)[:, None, None] * np.eye(2)
    psf.amplitudes = np.ones(npsf) / npsf
    return psf


def time_call(stmt, reps=5):
    '''
    Time a statement using the Python `timeit` module.

    `stmt` is a string containing the statement to be timed.
    '''

    # Grab the locals from the calling frame
    frame = inspect.currentframe()
    g = frame.f_back.f_locals.copy()
    g.update(globals())
    
    t = timeit.Timer(stmt, globals=g)

    # Determine the number of repetitions needed to get reliable timings
    n, total = t.autorange()

    # Now run that a few times to get a lower bound
    rep_timings = t.repeat(reps, n)
    return min(rep_timings)/n


# --- Time Specific Functions ---

def lnlike_test(nband=1, nx=16, ngal=1, nstamp=1, npsf=1):
    scene = get_scene(ngal, nx, nband=nband)
    stamps = get_stamps(nstamp, nx, npsf, nband=nband)
    plans = [WorkPlan(stamp) for stamp in stamps]
    theta = scene.get_all_source_params()
    t = time_call('lnlike_multi(theta, scene, plans)')
    return t

def plan_test(nx=16, ngal=1, npsf=1, nstamp=1, nband=1):
    scene = get_scene(ngal, nx, nband=nband)
    stamps = get_stamps(1, nx, npsf, nband=1)
    plan = WorkPlan(stamps[0])
    t = time_call('plan_sources(plan, scene)')
    return t


def convert_test(npsf=1, nx=16, **extras):
    scene = get_scene(1, nx, nband=1)
    stamps = get_stamps(1, nx, npsf, nband=1)
    t = time_call('convert_to_gaussians(scene.sources[0], stamps[0])')
    return t


def convert_grad_test(npsf=1, nx=16, **extras):
    scene = get_scene(1, nx, nband=1)
    stamps = get_stamps(1, nx, npsf, nband=1)
    gig = convert_to_gaussians(scene.sources[0], stamps[0])
    t = time_call('get_gaussian_gradients(scene.sources[0], stamps[0], gig)')
    return t


def compute_test(npsf=1, nx=16, **extras):
    scene = get_scene(1, nx, nband=1)
    stamps = get_stamps(1, nx, npsf, nband=1)
    gig = convert_to_gaussians(scene.sources[0], stamps[0])
    t = time_call('compute_gaussian(gig.gaussians[0,0], stamps[0].xpix.reshape(-1), stamps[0].ypix.reshape(-1), **extras)')
    return t * np.prod(gig.gaussians.shape)


if __name__ == "__main__":
    params = {"nband":  [1],
              "nx":     [16, 32, 64, 128],
              "ngal":   [1, 2, 4, 8, 12],
              "nstamp": [1, 2, 4, 8],
              "npsf":   [1, 2, 4, 8]
            }

    powers = np.arange(3,9)
    npix = 1. * 2**(powers*2)
    nx = 2**powers
    nstamp = np.array([1, 2, 4, 8])
    ngal = np.array([1, 2, 4, 8])
    npsf = np.array([1, 2, 4, 8, 16])

    ltimes = np.array([lnlike_test(nx=2**i) for i in powers])
    stimes = np.array([lnlike_test(nx=32, nstamp=i) for i in nstamp])
    gtimes = np.array([lnlike_test(nx=32, ngal=i) for i in ngal])
    ptimes = np.array([lnlike_test(nx=32, npsf=i) for i in npsf])

    #plantimes = [time_test.plan_test(nx=32, npsf=i) for i in npsf]
    #gradtimes = [time_test.convert_grad_test(nx=32, npsf=i) for i in npsf]

    conv, tunit = 1e3, "ms"
    
    import matplotlib.pyplot as pl
    os.makedirs('figures', exist_ok=True)

    nfig, nax = pl.subplots()
    nax.plot(npix,  npix*1.0/npix[-1] * ltimes[-1] * conv, ':', linewidth=3, label="linear")
    nax.plot(npix, ltimes * conv, '-o', label="forcepho")
    nax.set_xscale("log")
    nax.set_yscale("log")
    nax.set_xlabel("N$_{pix} / stamp$")
    nax.set_ylabel("likelihood call time ({})".format(tunit))
    nax.set_title("N$_{stamp} \\times N_{source} \\times N_{gauss}=10$")

    nax.plot(20*20, 5.5e-3 * conv, 'o', label="GALFIT", markersize=10)
    nax.fill_between(400 * np.array([0.9, 1.1]), np.zeros(2) + 1.5e-3*conv , np.zeros(2) + 8e-3*conv,
                    alpha=0.5, label="libprofit (R17)", color="tomato")
    nax.set_ylim(1e-1, 2e2)
    nax.legend()
    nfig.savefig(pjoin("figures","timing_npix.pdf"))

    t0 = np.mean([t[0] for t in [stimes, gtimes, ptimes]])
    N = np.linspace(1., 16., 100)

    sfig, sax = pl.subplots()
    sax.plot(N * 10,  t0 * N/N[0] * conv, ':', linewidth=3, label="linear")
    sax.plot(nstamp * 10, stimes * conv, '-o', label="vary N$_{stamp}$")
    sax.plot(ngal * 10, gtimes * conv, '-o', label="vary N$_{source}$")
    sax.plot(npsf * 10 , ptimes * conv, '-o', label="vary N$_{gauss}$")
    sax.set_xscale("log")
    sax.set_yscale("log")
    sax.set_xlabel("N$_{stamp} \\times N_{source} \\times N_{gauss}$")
    sax.set_ylabel("likelihood call time ({})".format(tunit))
    sax.set_title("N$_{{pix}} / stamp = ${:.0f}".format(32**2))
    sax.set_ylim(1, 2e2)
    sax.legend()
    sfig.savefig(pjoin("figures", "timing_ngauss.pdf"))


    # compare analytic to numerical derivative times
    npsf = 4
    ng = 9
    nx = 32
    # Analytic
    ta = (convert_test(npsf, nx) + convert_grad_test(npsf, nx) + compute_test(npsf, nx, compute_deriv=True))
    # Numerical (one calculation without gradients)
    tn = convert_test(npsf, nx) + compute_test(npsf, nx, compute_deriv=False)
    print(ta / tn)
