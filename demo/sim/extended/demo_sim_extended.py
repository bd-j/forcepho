import sys, os
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import Galaxy, Scene
from forcepho.likelihood import WorkPlan, make_image, lnlike_multi

from demo_utils import make_real_stamp as make_stamp
from demo_utils import Posterior


psfpaths = {"F090W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F115W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F150W": os.path.join(paths.psfmixture, 'f150w_ng6_em_random.p'),
            "F200W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            "F277W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            }

def prep_stamps(imnames, psfnames, ra_center, dec_center, size=(50, 50)):
    # HUUGE HAAAACK
    if filters[0] == "F277W":
        psfcenter = (496/2. - 100)
        psf_realization = 0
    else:
        psfcenter = 104.
        psf_realization = 2

    # --- Build the postage stamp ----
    pos = (ra_center, dec_center)
    stamps = [make_stamp(im, pos, center_type='celestial', size=size,
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


def prep_scene(sourcepars, filters=["dummy"], splinedata=None):

    # --- Get Sources and a Scene -----
    sources = []

    for pars in sourcepars:
        flux, x, y, q, pa, n, rh = np.copy(pars)
        if splinedata is not None:
            s = Galaxy(filters=filters, splinedata=splinedata)
            s.sersic = n
            s.rh = rh
        else:
            s = SimpleGalaxy(filters=filters)
        s.flux = flux
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()
    return scene, theta


def read_cat(catname):
    cols = ['id', 'ra', 'dec', 'a', 'b', 'pa', 'n', 'mag']
    usecols = [0, 1, 2, 5, 6, 7, 8, 9]
    dt = np.dtype([(c, np.float) for c in cols])
    cat = np.genfromtxt(catname, usecols=usecols, dtype=dt)
    return cat


def cat_to_sourcepars(catrow):
    ra, dec = catrow["ra"], catrow["dec"]
    q = np.sqrt(catrow["b"] / catrow["a"])
    pa = 90.0 - catrow["pa"]
    n = catrow["n"]
    S = np.array([[1/q, 0], [0, q]])
    rh = np.mean(np.dot(np.linalg.inv(S), np.array([catrow["a"], catrow["b"]])))
    return [ra, dec, q, pa, n, rh]


class PartialPosterior(Posterior):

    def complete_theta(self, theta);
        return theta


if __name__ == "__main__":


    filters = ["F090W"]
    sca = ["482"]
    exps = ["001"]

    # ------------------------------------
    # --- Choose a scene center and get some source guesses---
    catnames = [os.path.join(paths.galsims, '031718', "tri_gal_cat_{}.txt".format(s)) for s in sca]
    cat = read_cat(catnames[0])

    sceneid = 7
    gal = cat[cat["mag"] == 25.0][3 * sceneid]
    choose = (np.abs(cat["ra"] - gal["ra"]) < 1e-4) & (np.abs(cat["dec"] - gal["dec"]) < 2e-4)
    catscene = cat[choose]
    
    ra_ctr, dec_ctr = catscene["ra"].mean(), catscene["dec"].mean()

    # HACK!!!
    zp = 27.4525
    mag_offset = 0.4
    fluxes = [[10**(0.4 * (zp - s["mag"] - mag_offset))] for s in catscene]
    sourcepars = [tuple([flux] + cat_to_sourcepars(s)) for flux, s in zip(fluxes, catscene)]

    # --------------------------------
    # --- Setup Scene and Stamp(s) ---
    imnames = ['sim_cube_{}_{}_{}.slp.fits'.format(f, s, e) for f, s, e in zip(filters, sca, exps)]
    imnames = [os.path.join(paths.galsims, '031718', im) for im in imnames]
    psfnames = [psfpaths[f] for f in filters]
    sz = (40, 40)

    stamps = prep_stamps(imnames, psfnames, ra_ctr, dec_ctr, size=sz)
    plans = [WorkPlan(stamp) for stamp in stamps]
    scene, theta = prep_scene(sourcepars, filters=np.unique(filters).tolist(),
                              splinedata=paths.galmixture)

    theta_init = theta.copy()
    ptrue = theta.copy()
    ndim = len(theta)
    nsource = len(sourcepars)

    initial, _ = make_image(scene, stamps[0], Theta=theta_init)
    #sys.exit()

    # --------------------------------
    # --- Priors ---
    plate_scale = np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky))
    plate_scale = np.abs(plate_scale).mean()

    upper = [[12.0, s[1] + 3 * plate_scale, s[2] + 3 * plate_scale, 1.0, np.pi/2, 5.0, 0.12]
             for s in sourcepars]
    lower = [[0.0, s[1] - 3 * plate_scale, s[2] - 3 * plate_scale, 0.3, -np.pi/2, 1.0, 0.02]
             for s in sourcepars]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)

    # --------------------------------
    # --- Show model and data ---
    if True:
        fig, axes = pl.subplots(1, 2)
        axes[0].imshow(stamps[0].pixel_values.T, origin='lower')
        axes[1].imshow(initial.T, origin='lower')

    # --------------------------------
    # --- sampling ---
    if False:
        lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
        theta_width = (upper - lower)
        nlive = 50
        
        def prior_transform(unit_coords):
            # now scale and shift
            theta = lower + theta_width * unit_coords
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

        from dynesty import plotting as dyplot
        truths = ptrue.copy()
        label = filters + ["ra", "dec", "q", "pa", "n", "rh"]
        cfig, caxes = dyplot.cornerplot(results, fig=pl.subplots(ndim, ndim, figsize=(13., 10)),
                                        labels=label, show_titles=True, title_fmt='.8f', truths=truths)
        tfig, taxes = dyplot.traceplot(results, fig=pl.subplots(ndim, 2, figsize=(13., 13.)),
                                    labels=label)

        
    if True:
        p0 = ptrue.copy()
        scales = upper - lower
        scales = np.array(nsource * [ 10. ,   1. * plate_scale ,   1.*plate_scale ,   0.5,   3. ,   4. ,   0.1 ])
        #scales = np.array([ 50. ,   10. ,   10. ,   1.,   3. ,   4. ,   0.1 ])
        #scales = np.array([100., 5., 5., 1., 3., 5., 1.0])

        from hmc import BasicHMC
        model = Posterior(scene, plans, upper=upper, lower=lower)
        sampler = BasicHMC(model, verbose=False)
        sampler.ndim = len(p0)
        sampler.sourcepars = sourcepars
        sampler.stamps = stamps
        sampler.filters = filters
        sampler.offsets = None
        sampler.plans = plans
        sampler.scene = scene
        sampler.truths = ptrue.copy()

        sampler.set_mass_matrix(1/scales**2)
        eps = sampler.find_reasonable_stepsize(p0*1.0)
        use_eps = 0.1
        print(eps)
        #sys.exit()
        pos, prob, grad = sampler.sample(p0, iterations=10, mass_matrix=1/scales**2,
                                         epsilon=use_eps, length=20, sigma_length=5,
                                         store_trajectories=True)
        sys.exit()
        eps = sampler.find_reasonable_stepsize(pos)
        pos, prob, grad = sampler.sample(p0, iterations=1000, mass_matrix=1/scales**2,
                                         epsilon=eps/2, length=20, sigma_length=5,
                                         store_trajectories=True)
        #hresults = {"samples":sampler.chain.copy()}

        sys.exit()
        sampler.model = None
        import pickle
        with open("test_sersic_v2.pkl", "wb") as f:
            pickle.dump(sampler, f)
        sampler.model = model
