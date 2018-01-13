# This is all actually handled within sources.  leaving code here for easy testing.

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import h5py, sys

from scipy.interpolate import SmoothBivariateSpline
from scipy.special import gamma, gammainc

from gaussian_galaxy import sersic_profile, normal_oned
from gaussian_galaxy import plot_profiles, plot_cfd


def gauss_sersic_profile(x, amps, disps, smoothing=0.0):
    disps = np.hypot(disps, smoothing)
    a = amps / (disps * np.sqrt(2 * np.pi))
    gm = normal_oned(x, 0.0, a, disps)
    return gm


fn = "gauss_gal_results/sersic_mog_model.smooth=0.0075.h5"

with h5py.File(fn, "r") as data:
    n = data["nsersic"][:]
    r = data["rh"][:]
    A = data["amplitudes"][:]
    smoothing = data.attrs["smoothing"]
    radii = data["radii"][:]
    xx = data["x"][:]


nm, ng = A.shape
splines = [SmoothBivariateSpline(n, r, A[:, i], s=None) for i in range(ng)]

# --- look a the splines in rh for a given n ---
if True:

    def show_splines(ncheck=2.0):
        choose = (n == ncheck)
        cfig, caxes = pl.subplots(3, 3)
        cfig.suptitle('n={:3.1f}'.format(ncheck))
        for gind in range(ng):
            cax = caxes.flat[gind]
            cax.plot(r[choose], A[choose, gind], 'o', label='fitted')
            cax.plot(r[choose], np.squeeze(splines[gind](ncheck, r[choose])), label='spline')
        return cfig
    
    pdf = PdfPages('view_splines.pdf')
    for ncheck in np.unique(n):
        cfig = show_splines(ncheck)
        pdf.savefig(cfig)
        pl.close(cfig)
    pdf.close()

# --- check the amplitude sums ---
if False:
    def total_flux(ns=2.2, rh=0.07):
        amps = np.squeeze(np.array([spline(ns, rh) for spline in splines]))
        return amps.sum()
    ntest = 40
    ns_test = np.random.uniform(n.min(), n.max(), ntest)
    rh_test = np.random.uniform(r.min(), r.max(), ntest)
    for ns, rh in zip(ns_test, rh_test):
        print(ns, rh, total_flux(ns, rh))


# --- check the radial profiles between fitted values ---
if False:
    def test_splines(ns=2.2, rh=0.07):
        truth = sersic_profile(xx, n=ns, rh=rh, sigma=smoothing)
        amps = np.squeeze(np.array([spline(ns, rh) for spline in splines]))
        gm = gauss_sersic_profile(xx, amps, radii, smoothing=smoothing)
        gm_rh = gauss_sersic_profile(np.array([rh]), amps, radii, smoothing=smoothing)
        truth *= gm_rh[0]
        fig = plot_profiles(xx, truth, gm, ns=ns, rh=rh,
                            radii=radii, amps=amps,
                            smoothing=smoothing)

        return fig

    ntest = 20
    ns_test = np.random.uniform(n.min(), n.max(), ntest)
    rh_test = np.random.uniform(r.min(), r.max(), ntest)

    pdf = PdfPages('test_splines.pdf')
    for ns, rh in zip(ns_test, rh_test):
        fig = test_splines(ns, rh)
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()


