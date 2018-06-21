import sys, os, time
from functools import partial as argfix
import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import Galaxy, Scene
from forcepho.likelihood import WorkPlan, make_image, lnlike_multi


from demo_utils import make_stamp, negative_lnlike_multi, numerical_image_gradients, Posterior


def setup_scene(sourceparams=[(1.0, 5., 5., 0.7, 30., 1.0, 0.05)],
                splinedata=None, perturb=0,
                filters=['dummy'],
                plate_scale=0.032,
                add_noise=False, snr_max=100.,
                
                stamp_kwargs=[]):

    # --- Get Sources and a Scene -----
    sources = []
    for (flux, x, y, q, pa, n, rh) in sourceparams:
        s = Galaxy(filters=filters, splinedata=splinedata)
        s.sersic = n
        s.rh = rh
        s.flux = flux
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()
    label = []

    # get stamps
    stamps = [make_stamp(**sk) for sk in stamp_kwargs]
    for stamp in stamps:
        stamp.scale /= plate_scale
    
    # --- Generate mock and add to stamp ---
    ptrue = theta * np.random.normal(1.0, perturb, size=theta.shape)
    for stamp in stamps:
        true_image, _ = make_image(scene, stamp, Theta=ptrue)
        stamp.pixel_values = true_image.copy()
        err = stamp.pixel_values.max() / snr_max
        #err = np.sqrt(err**2 + stamp.pixel_values.flatten())
        err = np.ones(stamp.npix) * err
        stamp.ierr = np.ones(stamp.npix) / err
        if add_noise:
            noise = np.random.normal(0, err)
            stamp.pixel_values += noise.reshape(stamp.nx, stamp.ny)


    label = filters + ["ra", "dec", "q", "pa", "n", "rh"]

    return scene, stamps, ptrue, label


if __name__ == "__main__":

    # --- Setup Scene and Stamp(s) ---

    # Let's make one stamp
    filters = ["F090W"] #, "F150W"]
    psfnames = ['f090_ng6_em_random.p'] # , 'f150w_ng6_em_random.p']
    psfnames = [os.path.join(paths.psfmixture, p) for p in psfnames]
    stamp_kwargs = [{'size': (50, 50), 'psfname': p, 'filtername': f}
                    for p, f in zip(psfnames, filters)]

    # Let's make one Galaxy in one band
    # flux, ra, dec, q, pa(deg), n, sersic
    sourcepars = [([30.], 25., 25., 0.7, 45, 2.1, 0.07)]
    upper = np.array([60., 30., 30., 1.0, np.pi/2., 5.0, 0.12])
    lower = np.array([10., 20., 20., 0.0, -np.pi/2, 1.0, 0.03])

    scene, stamps, ptrue, label = setup_scene(sourceparams=sourcepars,
                                              splinedata=paths.galmixture,
                                              perturb=0.0, add_noise=True,
                                              snr_max=20., filters=filters,
                                              plate_scale=0.032,
                                              stamp_kwargs=stamp_kwargs)

    Theta = ptrue.copy()
    ndim = len(ptrue)
    plans = [WorkPlan(stamp) for stamp in stamps]
    nll, nll_grad = negative_lnlike_multi(Theta, scene=scene, plans=plans)

    # --- Image gradients ---
    if False:
        delta = np.ones_like(ptrue) * 1e-7
        #numerical
        grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
        image, grad = make_image(scene, stamps[0], Theta=ptrue)
        fig, axes = pl.subplots(len(ptrue), 3, sharex=True, sharey=True)
        for i in range(len(ptrue)):
            g = grad[i,:].reshape(stamp.nx, stamp.ny)
            c = axes[i, 0].imshow(grad_num[i,:,:].T, origin='lower')
            fig.colorbar(c, ax=axes[i, 0])
            c = axes[i, 1].imshow(g.T, origin='lower')
            fig.colorbar(c, ax=axes[i, 1])
            c = axes[i, 2].imshow((grad_num[i,:,:] - g).T, origin='lower')
            fig.colorbar(c, ax=axes[i, 2])

        axes[0, 0].set_title('Numerical')
        axes[0, 1].set_title('Analytic')
        axes[0, 2].set_title('N - A')
        pl.show()

    # --- Slices ---
    if False:

        # Rh
        npg = 20
        lo, hi = scene.sources[0].rh_range
        rgrid = np.linspace(lo, hi, npg)
        lo, hi = scene.sources[0].sersic_range
        ngrid = np.linspace(lo, hi, npg)
        lo, hi = lower[0], upper[0]
        fgrid = np.linspace(lo, hi, npg)
        
        grids = [rgrid, ngrid, fgrid]
        ind = [-1, -2, 0]
        p = ["$R_h$", "$n_{sersic}$", "flux"]

        xi, dxi = [], []

        fig, axes = pl.subplots(2, len(grids))
        for g, grid in enumerate(grids):
            x, del_x = [], []
            for i, v in enumerate(grid):
                pnew = ptrue.copy()
                pnew[ind[g]] = v
                nll, nll_grad = negative_lnlike_multi(pnew, scene=scene, plans=plans)
                x.append(nll)
                del_x.append(nll_grad)
            xi.append(x)
            dxi.append(del_x)
            dxi_dp = np.gradient(x) / np.gradient(grid)
            axes[0, g].plot(grid, x, '-o')
            axes[1, g].plot(grid, np.array(del_x)[:, ind[g]], '-o')
            axes[1, g].plot(grid, dxi_dp, '-o')
            axes[1, g].set_xlabel(p[g])
        axes[0, 0].set_ylabel("$\\chi^2/2$")
        axes[1, 0].set_ylabel("$\partial\\chi^2/\partial \\theta$")


        pl.show()

    # ------------------
    # --- sampling

    # --- hemcee ---
    if False:
        p0 = ptrue.copy()
        scales = upper - lower
        scales = np.array([ 5. ,   0.2 ,   0.2 ,   0.1,   0.2,   1.0 ,   0.02 ])

        from hemcee import NoUTurnSampler
        from hemcee.metric import DiagonalMetric
        metric = DiagonalMetric(scales**2)
        model = Posterior(scene, plans, upper=upper, lower=lower)
        sampler = NoUTurnSampler(model.lnprob, model.lnprob_grad, metric=metric)


        t = time.time()
        pos, lnp0 = sampler.run_warmup(p0, 600)
        twarm = time.time() - t
        ncwarm = np.copy(model.ncall)
        model.ncall = 0
        t = time.time()
        chain, lnp = sampler.run_mcmc(pos, 2000)
        tsample = time.time() - t
        nc = model.ncall

    # --- nested ---
    if False:
        lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
        theta_width = (upper - lower)
        nlive = 50
        
        def prior_transform(unit_coords):
            # now scale and shift
            theta = lower + theta_width * unit_coords
            return theta

        import dynesty
        
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
                                        labels=label, show_titles=True, title_fmt='.3f',
                                        title_kwargs={'fontsize':12}, truths=truths)
        tfig, taxes = dyplot.traceplot(results, fig=pl.subplots(ndim, 2, figsize=(13., 13.)),
                                    labels=label)

    # --- hmc ----
    if True:
        p0 = ptrue.copy()
        scales = upper - lower
        scales = np.array([ 1. ,   0.05 ,   0.05 ,   0.03,   0.05,   0.1 ,   0.005 ])

        from hmc import BasicHMC
        model = Posterior(scene, plans, upper=upper, lower=lower, verbose=False)
        sampler = BasicHMC(model, verbose=False)
        sampler.ndim = len(p0)
        sampler.sourcepars = sourcepars
        sampler.stamp_kwargs = stamp_kwargs
        sampler.filters = filters
        sampler.offsets = None
        sampler.plans = plans
        sampler.scene = scene
        sampler.truths = ptrue.copy()

        sampler.set_mass_matrix(1/scales**2)
        eps = sampler.find_reasonable_stepsize(p0)
        # eps = 0.00390625
        #sys.exit()
        t = time.time()
        pos, prob, grad = sampler.sample(p0, iterations=10, mass_matrix=1/scales**2,
                                         epsilon=eps/2, length=20, sigma_length=5,
                                         store_trajectories=True)
        ncburn = np.copy(model.ncall)
        model.ncall = 0
        tburn = time.time() - t
        eps = sampler.find_reasonable_stepsize(pos)
        t = time.time()
        pos, prob, grad = sampler.sample(p0, iterations=1000, mass_matrix=1/scales**2,
                                         epsilon=eps/2, length=20, sigma_length=5,
                                         store_trajectories=True)
        thmc = time.time() - t
        #results = {"samples":sampler.chain.copy()}
        
        sys.exit()
        sampler.model = None
        import pickle
        with open("test_sersic_v2.pkl", "wb") as f:
            pickle.dump(sampler, f)
        sampler.model = model
