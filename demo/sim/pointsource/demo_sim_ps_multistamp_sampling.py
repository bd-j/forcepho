# ----------
# Script to fit a single point source in multiple (same band) simulated images.
#-----------

import sys, os
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import Star, Scene
from forcepho.likelihood import WorkPlan, lnlike_multi, make_image

from demo_utils import make_real_stamp as make_stamp


imnames = ['sim_cube_F090W_487_001.slp.fits', 'sim_cube_F090W_487_008.slp.fits']
imnames = [os.path.join(paths.starsims, im) for im in imnames]
psfname = os.path.join(paths.psfmixture, 'f090_ng6_em_random.p')


def fit_source(ra=53.115295, dec=-27.803501, dofit=True, nlive=100):

    # --- Build the postage stamp ----
    ra_init, dec_init = ra, dec
    pos_init = (ra_init, dec_init)
    stamps = [make_stamp(im, pos_init, center_type='celestial', size=(50, 50), psfname=psfname)
              for im in imnames]

    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    for s in stamps:
        s.psf.covariances = np.matmul(T, np.matmul(s.psf.covariances, T.T))
        s.psf.means = np.matmul(s.psf.means, T)

    # --- get the Scene ---    
    source = Star(filters=["F090W"])
    scene = Scene([source])
    label = ['Counts', 'RA', 'Dec']

    plans = [WorkPlan(stamp) for stamp in stamps]
    lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)

    # --- Initialize ---
    theta_init = np.array([stamps[0].pixel_values.sum() * 1.0, ra_init, dec_init])
    # a rough measure of dcoordinate/dpix
    plate_scale, _ = np.linalg.eig(np.linalg.inv(stamps[0].dpix_dsky))
    # make the prior ~10 pixels wide, and 50% of counts
    theta_width = np.array([0.5 * theta_init[0], 10 * plate_scale[0], 10 * plate_scale[1]])
    #print(theta_init, theta_width)

    # --- Nested sampling ---
    ndim = 3

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


    # ---- Read the input catalog -----
    catname = os.path.join(paths.starsims, 'stars_f090w.cat')
    dt = np.dtype([(n, np.float) for n in ['ra', 'dec', 'x', 'y', 'mag', 'counts', 'flux1', 'flux2']])
    cat = np.genfromtxt(catname, usecols=np.arange(1, 9), dtype=dt)
    cat = cat[:100]

    # --- setup output ---
    #fn = 'output_pointsource_dynesty.dat'
    #pn = 'pointsource_resid.pdf'
    #out = open(fn, 'w')
    #strfmt = "{}  {:11.8f}   {:11.8f}  {:10.2f}  {:10.2f}  {:14.6f}   {} \n"
    #dt2 = np.dtype([(n, np.float) for n in ['id', 'ra', 'dec', 'counts', 'sum', 'lnp', 'ncall']])
    #out.write("# i    ra    dec   counts    imsum    lnp   ncall\n")
    #pdf = PdfPages(pn)

    label = ['Counts', 'RA', 'Dec']

    all_results, all_pos = [], []
    for i, c in enumerate(cat[:1]):
        blob = fit_source(ra=c['ra'], dec=c['dec'], nlive=100)

        result, vals, stamps, scene = blob
        all_results.append(result)

        #from dynesty import plotting as dyplot
        #cfig, caxes = dyplot.cornerplot(result, fig=pl.subplots(3,3, figsize=(13., 10)),
        #                                labels=label, show_titles=True, title_fmt='.8f')
        #tfig, taxes = dyplot.traceplot(result, fig=pl.subplots(3, 2, figsize=(13., 13.)),
        #                               labels=label)
        #
        #rfig, raxes = pl.subplots(len(stamps), 3, sharex=True, sharey=True)
        #for i, stamp in enumerate(stamps):
        #    im, grad = make_image(scene, stamp, Theta=vals)
        #    raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
        #    raxes[i, 1].imshow(im.T, origin='lower')
        #    resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
        #    rfig.colorbar(resid, ax=raxes[i,:].tolist())
        #[ax.set_xlim(35, 65) for ax in raxes.flat]
        #[ax.set_ylim(35, 65) for ax in raxes.flat]
        #pdf.savefig(rfig)
        #pl.close(rfig)

        print('----')
        print(vals)
        print(c['counts'])

        counts = vals[0]
        center = np.array(stamp.pixcenter_in_full)
        lo = (center - 0.5 * size).astype(int)
        x, y = lo + vals[1:]
        all_pos.append((x, y))

        out.write(strfmt.format(i, x, y, counts, result['logl'].max(), stamp.pixel_values.sum(), result['ncall']))

    out.close()
    #ocat = np.genfromtxt(fn, dtype=dt2)
    
