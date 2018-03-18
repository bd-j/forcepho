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
from demo_utils import Posterior


imnames = ['sandro/sim_F090W_482_star_grid.slp.fits']
imnames = [os.path.join(paths.starsims, im) for im in imnames]
psfnames = ['f090_ng6_em_random.p']
psfnames = [os.path.join(paths.psfmixture, p) for p in psfnames]
filters = ["F090W"]
catname = os.path.join(paths.starsims, 'sandro/star_cat_482.txt')


def prep_stamps(ra, dec):
    # HUUGE HAAAACK
    if filters[0] == "F277W":
        psfcenter = (496/2. - 100)
        psf_realization = 0
    else:
        psfcenter = 104.
        psf_realization = 2

    # --- Build the postage stamp ----
    ra_init, dec_init = ra, dec
    pos_init = (ra_init, dec_init)
    stamps = [make_stamp(im, pos_init, center_type='celestial', size=(50, 50),
                         psfname=pn, psfcenter=psfcenter, fix_header=True,
                         psf_realization=psf_realization)
              for im, pn in zip(imnames, psfnames)]


    # Background subtract.  yuck
    for stamp in stamps:
        bkg = np.nanmedian(stamp.pixel_values[:20, :])  # stamp.full_header["BKG"]
        stamp.pixel_values -= bkg # 
        stamp.subtracted_background = bkg
        
    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    for s in stamps:
        s.psf.covariances = np.matmul(T, np.matmul(s.psf.covariances, T.T))
        s.psf.means = np.matmul(s.psf.means, T)

    return stamps


def fit_source(ra=53.115295, dec=-27.803501, mag=None, dofit='dynesty', nlive=100):

    # --- Get the data ---
    stamps = prep_stamps(ra, dec)

    # --- Get the Scene ---    
    source = Star(filters=filters)
    scene = Scene([source])
    label = ['Counts', 'RA', 'Dec']

    plans = [WorkPlan(stamp) for stamp in stamps]

    # --- Initialize and set scales ---
    if mag is None:
        counts = [np.clip(stamp.pixel_values.sum(), 1, np.inf) for stamp in stamps]
    else:
        counts = [10**(0.4 * (stamp.full_header["ABMAG"] - mag)) for stamp in stamps]
    theta_init = np.array(counts + [ra, dec])
    # a rough measure of dcoordinate/dpix - this doesn't work so great
    plate_scale = np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky))
    plate_scale = np.abs(plate_scale)
    # make the prior ~5 pixels wide, and 100% of expected counts
    theta_width = np.array([theta_init[0], 5 * plate_scale[0], 5 * plate_scale[1]])
    print(theta_init, theta_width)

    # --- Sampling ---
    ndim = 3

    if dofit == 'dynesty':

        lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
        def prior_transform(unit_coords):
            # convert to uniform -1 to 1
            u = (2 * unit_coords - 1.)
            # now scale and shift
            theta = theta_init + theta_width * u
            return theta

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

    elif dofit == "hmc":

        p0 = theta_init.copy()
        upper = theta_init + theta_width / 2.
        lower = theta_init - theta_width / 2.
        scales = theta_width
        
        model = Posterior(scene, plans, upper=upper, lower=lower)
        sampler = BasicHMC(model, verbose=False)
        sampler.ndim = len(p0)
        pos, prob, grad = sampler.sample(p0, iterations=100, mass_matrix=1/scales**2,
                                         epsilon=None, length=20, sigma_length=5,
                                         store_trajectories=True)
        results = {"samples":sampler.chain.copy()}

    else:
        results = None
        theta_max = np.zeros(3)

    return results, theta_max, stamps, scene


if __name__ == "__main__":

    showfit = True
    start = 30
    end = 100
    nlive = 80

    # ---- Read the input catalog -----
    dt = np.dtype([(n, np.float) for n in ['id', 'ra', 'dec', 'mag']])
    cat = np.genfromtxt(catname, usecols=(0, 1, 2, 3), dtype=dt)

    # --- setup output ---
    nband = len(filters)
    fn = 'output_pointsource_sandro_{}.dat'.format(filters[0])
    pn = 'pointsource_resid_sandro_{}.pdf'.format(filters[0])
    out = open(fn, 'w')
    strfmt = "{}  {:11.8f}   {:11.8f}" + "".join((nband * 3) * ["  {:10.2f}"]) + "  {:10.2f}  {:14.6f} {} \n"
    cols = ("# i    ra    dec" +
            "".join(["  counts16_{f}   counts50_{f}   counts84_{f}".format(f=f.lower()) for f in filters]) +
            "  imsum    lnp   ncall\n")
    out.write(cols)
    if showfit:
        pdf = PdfPages(pn)

    label = ['Counts', 'RA', 'Dec']

    # --- Loop over sources ---
    all_results, all_pos = [], []
    for i, c in enumerate(cat[start:end]):
        blob = fit_source(ra=c['ra'], dec=c['dec'], mag=c["mag"], nlive=nlive, dofit='dynesty')

        result, vals, stamps, scene = blob
        all_results.append(result)
        ndim = len(vals)

        if showfit:
            sz = 6 * 3 + 2, len(stamps) * 2.5  # figure size
            rfig, raxes = pl.subplots(len(stamps), 6, sharex=True, sharey=True, figsize=sz)
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
            raxes[0,0].text(0.6, 0.75, "id={:3.0f}\nmag={:2.1f}".format(c["id"], c["mag"]), transform=raxes[0,0].transAxes)
            pdf.savefig(rfig)
            pl.close(rfig)

        print('----')
        print(vals)
        print(c['mag'], c['ra'], c['dec'])

        counts = vals[0]
        size = np.array([stamp.nx, stamp.ny])
        center = np.array(stamp.pixcenter_in_full)
        lo = (center - 0.5 * size).astype(int)
        x, y = lo + vals[nband:]
        all_pos.append((x, y))
        wght = np.exp(result["logwt"] - result['logz'][-1])
        flux = np.array([quantile(result["samples"][:, k], [0.16, 0.5, 0.84], wght) for k in range(nband)])

        # cols = ([i] + vals[nband:].tolist() + vals[:nband].tolist() +
        cols = ([c["id"]] + vals.tolist()[nband:] + flux.flatten().tolist() + 
                [stamp.pixel_values.sum(), result['logl'].max(), result['ncall'].sum()])
        out.write(strfmt.format(*cols))

    if showfit:
        pdf.close()
    else:
        pl.show()
    out.close()
    
    ocat = np.genfromtxt(fn)#, dtype=dt2)
    inds = ocat[:,0].astype(int) -1
    mag = cat[inds]["mag"]
    model_mag = stamps[0].full_header["ABMAG"] - 2.5 * np.log10(ocat[:, 4])
    dm = mag - model_mag
    # This is not actually correct (need to use the actual stamp, not just the last one)
    pix = np.array([stamps[0].sky_to_pix(np.array([c["ra"], c["dec"]])) for c in cat[inds]])
    subpix = np.mod(pix, 1.0)
    
    sigma_m = 1.086 * 0.5 * (ocat[:, 5] - ocat[:, 3]) / ocat[:, 4]
    merr_up = 1.086 * (ocat[:, 5] - ocat[:, 4]) / ocat[:, 4]
    merr_lo = 1.086 * (ocat[:, 4] - ocat[:, 3]) / ocat[:, 4]
    

    dsky = np.array([ocat[:, 1] - cat[inds]["ra"], ocat[:, 2] - cat[inds]["dec"]])
    dpix = np.dot(stamps[0].dpix_dsky, dsky)  # This is not actually correct (need to use the actual stamp, not just the last one)

    sigma_scale = [((dm[i*10:i*10+10]-dm[i*10:i*10+10].mean())/sigma_m[i*10:i*10+10]).std() for i in range(7)]
    mscatter = [((dm[i*10:i*10+10]-dm[i*10:i*10+10].mean())).std() for i in range(7)]
    sigma_median = [np.median(sigma_m[i*10:i*10+10]) for i in range(7)]

    efig, eax = pl.subplots()
    eax.errorbar(mag, dm, [merr_lo, merr_up], capsize=6, marker='o', capthick=1, linestyle='')
    eax.set_xlabel("AB mag (input)")
    eax.set_ylabel("$\Delta$ m (Input - Output)")
    eax.set_xlim(21.5, 28.5)
    
    sys.exit()
    from dynesty import plotting as dyplot
    i = -11
    result = all_results[i]
    c = cat[inds[i]]
    truths = [10**(0.4 * (stamp.full_header["ABMAG"] - c["mag"])) for stamp in stamps] + [c["ra"] , c["dec"]]
    cfig, caxes = dyplot.cornerplot(result, fig=pl.subplots(ndim, ndim, figsize=(13., 10)),
                                    labels=label, show_titles=True, title_fmt='.8f', truths=truths)
    tfig, taxes = dyplot.traceplot(result, fig=pl.subplots(ndim, 2, figsize=(13., 13.)),
                                   labels=label)

    
