# ----------
# Script to fit a single point source in multiple (same band) simulated images.
#-----------

import sys, os
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from forcepho import paths
from forcepho.sources import Star, Scene
from forcepho.likelihood import WorkPlan, lnlike_multi, make_image

from corner import quantile

from demo_utils import make_real_stamp as make_stamp


imnames = ['sim_cube_F090W_487_001.slp.fits', 'sim_cube_F090W_487_008.slp.fits',
           'sim_cube_F150W_487_002.slp.fits', 'sim_cube_F150W_487_008.slp.fits']
imnames = [os.path.join(paths.starsims, im) for im in imnames]
psfnames = ['f090_ng6_em_random.p', 'f090_ng6_em_random.p',
            'f150w_ng6_em_random.p', 'f150w_ng6_em_random.p']
psfnames = [os.path.join(paths.psfmixture, p) for p in psfnames]
filters = ["F090W", "F150W"]


def fit_source(ra=53.115295, dec=-27.803501, dofit=True, nlive=100):

    # --- Build the postage stamp ----
    ra_init, dec_init = ra, dec
    pos_init = (ra_init, dec_init)
    stamps = [make_stamp(im, pos_init, center_type='celestial', size=(50, 50), psfname=pn)
              for im, pn in zip(imnames, psfnames)]

    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    for s in stamps:
        s.psf.covariances = np.matmul(T, np.matmul(s.psf.covariances, T.T))
        s.psf.means = np.matmul(s.psf.means, T)

    # --- get the Scene ---    
    source = Star(filters=filters)
    scene = Scene([source])
    label = ['Counts', 'RA', 'Dec']

    plans = [WorkPlan(stamp) for stamp in stamps]
    lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)

    # --- Initialize ---
    counts = [stamp[0].pixel_values.sum() * 1.0 for stamp in stamps]
    theta_init = np.array(counts + [ra_init, dec_init])
    # a rough measure of dcoordinate/dpix
    plate_scale, _ = np.linalg.eig(np.linalg.inv(stamps[0].dpix_dsky))
    # make the prior ~10 pixels wide, and 100% of counts
    theta_width = np.array(counts + [10 * plate_scale[0], 10 * plate_scale[1]])
    #print(theta_init, theta_width)

    # --- Nested sampling ---
    ndim = len(theta_init)

    def prior_transform(unit_coords):
        # convert to uniform -1 to 1
        u = (2 * unit_coords - 1.)
        # now scale and shift
        theta = theta_init + theta_width * u
        return theta

    if dofit:
        import dynesty, time
        
        # "Standard" nested sampling.
        sampler = dynesty.NestedSampler(lnlike, prior_transform, ndim, nlive=nlive, bootstrap=0)
        t0 = time.time()
        sampler.run_nested()
        dur = time.time() - t0
        results = sampler.results
        results['duration'] = dur
        indmax = results['logl'].argmax()
        theta_max = results['samples'][indmax, :]

    else:
        results = None
        theta_max = np.zeros(3)
        stamps = None

    return results, theta_max, stamps, scene

if __name__ == "__main__":


    showfit = True
    nfit = 20
    nband = len(filters)

    # ---- Read the input catalog(s) -----
    dt = np.dtype([(n, np.float) for n in ['ra', 'dec', 'x', 'y', 'mag', 'counts', 'flux1', 'flux2']])
    catnames = [os.path.join(paths.starsims, 'stars_{}.cat'.format(band.lower())) for band in filters]
    cats = [np.genfromtxt(cn, usecols=np.arange(1, 9), dtype=dt)[:100] for cn in catnames]
    
    # --- setup output ---    
    fn = 'output_pointsource_dynesty.dat'
    pn = 'pointsource_resid_dynesty.pdf'
    out = open(fn, 'w')
    strfmt = "{}  {:11.8f}   {:11.8f}" + "".join((nband * 3) * ["  {:10.2f}"]) + "  {:10.2f}  {:14.6f} {} \n"
    cols = ("# i    ra    dec" +
            "".join(["  counts16_{f}   counts50_{f}   counts84_{f}".format(f=f.lower()) for f in filters]) +
            "  imsum    lnp   ncall\n")
    # dt2 = np.dtype([(n, np.float) for n in ['id', 'ra', 'dec', 'counts', 'sum', 'lnp', 'ncall']])
    out.write(cols)
    if showfit:
        pdf = PdfPages(pn)

    label = ['Counts_{}'.format(f.lower()) for f in filters] + ['RA', 'Dec']

    #sys.exit()
    all_results, all_pos = [], []
    for i, c in enumerate(cats[0][:nfit]):
        blob = fit_source(ra=c['ra'], dec=c['dec'], nlive=50)
        result, vals, stamps, scene = blob
        all_results.append(result)

        #from dynesty import plotting as dyplot
        #cfig, caxes = dyplot.cornerplot(result, fig=pl.subplots(4,4, figsize=(13., 10)),
        #                                labels=label, show_titles=True, title_fmt='.8f')
        #tfig, taxes = dyplot.traceplot(result, fig=pl.subplots(4, 2, figsize=(13., 13.)),
        #                               labels=label)

        if showfit:
            rfig, raxes = pl.subplots(len(stamps), 6, sharex=True, sharey=True)
            raxes = np.atleast_2d(raxes)
            for j, stamp in enumerate(stamps):
                err = 1. / stamp.ierr.reshape(stamp.nx, stamp.ny)
                snr = stamp.pixel_values * stamp.ierr.reshape(stamp.nx, stamp.ny)
                model, grad = make_image(scene, stamp, Theta=vals)
                data = stamp.pixel_values
                resid = (data - model)
                chi = resid / err
                imlist = [err, snr, data, model, resid, chi]
                for ax, im in zip(raxes[j, :], imlist):
                    cc = ax.imshow(im[20:35, 20:35].T, origin='lower')
                rfig.colorbar(cc, ax=raxes[j,:].tolist())
            #[ax.set_xlim(35, 65) for ax in raxes.flat]
            #[ax.set_ylim(35, 65) for ax in raxes.flat]
            pdf.savefig(rfig)
            pl.close(rfig)

        print('----')
        print(vals)
        counts = [c[i]["counts"] for c in  cats]
        print(counts, c['ra'], c['dec'])

        #size = np.array([stamp.nx, stamp.ny])
        #center = np.array(stamp.pixcenter_in_full)
        #lo = (center - 0.5 * size).astype(int)
        #x, y = lo + vals[nband:]
        #all_pos.append((x, y))
        wght = np.exp(result["logwt"] - result['logz'][-1])
        flux = np.array([quantile(result["samples"][:, k], [0.16, 0.5, 0.84], wght) for k in range(nband)])
        cols = ([i] + vals.tolist()[nband:] + flux.flatten().tolist() + 
                [result['logl'].max(), stamp.pixel_values.sum(), result['ncall'].sum()])
        out.write(strfmt.format(*cols))

    if showfit:
        pdf.close()
    else:
        pl.show()
    out.close()

    
    ocat = np.genfromtxt(fn)#, dtype=dt2)
    ratio = ocat[:, 4] / cats[0][:nfit]["counts"]
    hi = ocat[:, 5]/cats[0][:nfit]["counts"] - ocat[:, 4]/cats[0][:nfit]["counts"]
    lo = ocat[:, 4]/cats[0][:nfit]["counts"] - ocat[:, 3]/cats[0][:nfit]["counts"]
    fig, ax = pl.subplots()
    ax.errorbar(cats[0][:nfit]["counts"], ratio, yerr=[lo, hi], capsize=5, capthick=2)
