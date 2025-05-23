#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fitting.py

Routines for fitting a Scene model to data, using older CPU-based
likelihood methods.
"""

import sys, os, time
from functools import partial as argfix
import numpy as np

from .model import ConstrainedTransformedPosterior as Posterior
from .utils import make_statscat, make_chaincat, get_sample_cat, extract_block_diag
from .region import CircularRegion


__all__ = ["Result", "run_lmc", "run_opt",
           "run_pymc3", "run_dynesty",
           "optimize_fluxes", "optimize_one_band"]


class Result(object):
    """A simple namespace for storing information about a run.
    """

    def __init__(self, filename=None, **kwargs):
        if filename:
            self.read_from_h5(filename)
            try:
                self._reconstruct()
            except:
                pass
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fill(self, region, active, fixed, model, big=None,
             bounds=None, patchID=None, step=None, stats=None):
        """
        Parameters
        ----------
        region : a region.Region object
            The region defining the patch; its parameters will be added to the
            result.

        active : structured ndarray
            The active sources and their starting parameters.

        fixed : structured ndarray
            The fixed sources and their parameters.

        model : a model.PosteriorModel object
            Must contain `patch` and `scene` attributes

        bounds : optional
            If given, a structured ndarrray of lower and upper bounds for
            each parameter of each source.

        patchID : optional
            An integer giving the unique patch ID.

        step : optional
            If supplied, a littlemcmc NUTS step obect.  this contains the covariance matrix

        stats : optional
            If supplied, a littlemcmc stats object.

        Returns
        -------
        qcat : structured ndarray
            A structured array of the parameter values in the last sample of the chain.

        block_covs : ndarray of shape (N_source, N_param, N_param)
            The covariance matrices for the sampling potential, extracted as block
            diagonal elements of the full N_source * N_param square covariance
            array.  Note that it is in the units of the transformed, unconstrained
            sampling parameters.  If the prior bounds change, the covariance matrix
            is no longer valid (or must be retransformed)
        """
        # shortcuts
        patch = model.patch
        scene = model.scene
        bands = np.array(patch.bandlist, dtype="S")  # Bands actually present in patch
        shapenames = np.array(scene.sources[0].SHAPE_COLS, dtype="S")
        ref = np.array(patch.patch_reference_coordinates)
        block_covs = None

        # --- Header ---
        self.patchID = patchID
        self.reference_coordinates = ref
        self.bandlist = bands  # FIXME: keep bandlist a list? It gets converted to list when reading...
        self.shapenames = shapenames

        # --- basic info ---
        self.n_call = model.ncall
        self.npix = patch.npix
        self.nexp = len(patch.epaths)
        self.exposures = np.array(patch.epaths, dtype="S")

        # --- region, active, fixed ---
        for k, v in region.__dict__.items():
            setattr(self, k, v)
        self.active = np.array(active)
        if fixed is not None:
            self.fixed = np.array(fixed)
        else:
            self.fixed = None
        if big is not None:
            self.big = np.array(big)
        else:
            self.big = None

        # --- chain as structured array ---
        if hasattr(self, "chain"):
            self.chaincat = make_chaincat(self.chain, self.bandlist, self.active,
                                          self.reference_coordinates, shapes=self.shapenames)

        # --- covariance ---
        if step is not None:
            try:
                self.cov = np.array(step.potential._cov.copy())
            except(AttributeError):
                self.cov = np.diag(step.potential._var.copy())
            if stats is not None:
                self.stats = make_statscat(stats, step)
                self.n_tune = np.sum(self.stats["tune"])
                self.n_sample = len(self.stats) - self.n_tune

            n_param = len(bands) + len(shapenames)
            block_covs = extract_block_diag(self.cov, n_param)

        # --- priors ---
        if bounds is not None:
            self.bounds = bounds

        # --- last position as structured array ---
        # FIXME: do this another way; use get_sample_cat with iteration = -1
        #qlast = self.chain[-1, :]
        #scene.set_all_source_params(qlast)
        #patch.unzerocoords(scene)
        #for i, source in enumerate(scene.sources):
        #    source.id = active[i]["source_index"]
        #qcat = scene.to_catalog(extra_cols=["source_index"])
        #qcat["source_index"][:] = active["source_index"]
        #self.final = qcat
        qcat = self.get_sample_cat(-1)
        self.final = qcat

        return qcat, block_covs

    def get_sample_cat(self, iteration):
        """Get a sample of the scene parameters from the chain, as a structured
        arrays with one row for each source.

        Parameters
        ----------
        iteration : int
            The iteration of the chain for which to produce a catalog.

        Returns
        -------
        sample : structured ndarray of shape (n_active,)
            The parameters of each source at the specified iteration of the
            chain, as a structured array.
        """
        sample = get_sample_cat(self.chaincat, iteration, self.active)
        return sample

    def get_map(self, structured=False):
        """Get the highest probability sample

        Returns
        -------
        ymap : ndarray of shape (1, n_dim) by default or structured ndararry of length (n_active,)
        """
        lnp = self.stats["model_logp"]
        ind_ml = np.argmax(lnp)
        if structured:
            ymap = self.get_sample_cat(ind_ml)
        else:
            ymap = np.atleast_2d(self.chain[ind_ml, :])

        return ymap

    def dump_to_h5(self, filename, verbose=False):
        """Try to save attributes of the Result() as satasets or attributes in
        the given HDF5 file.
        """
        msgs = []
        import h5py
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, "w") as out:
            for name, value in vars(self).items():
                if type(value) == np.ndarray:
                    out.create_dataset(name, data=value)
                else:
                    try:
                        out.attrs[name] = value
                    except:
                        msg = "could not save {} with value {} to {}".format(name, value, filename)
                        if verbose:
                            print(msg)
                        else:
                            msgs.append(msg)

        return msgs

    def read_from_h5(self, filename):
        """Read attributes of the Results() object form an HDF5 file.
        """
        import h5py
        out = self
        with h5py.File(filename, "r") as f:
            for k, v in f.items():
                setattr(out, k, f[k][:])
            for k, v in f.attrs.items():
                setattr(out, k, v)

    def _reconstruct(self):
        """Convert some string types to utf-8, and make a structured array
        chaincat attribute from the raw parameter chain.
        """
        self.bands = [b.decode("utf") for b in self.bandlist]
        self.shape_cols = [s.decode("utf") for s in self.shapenames]
        self.region = CircularRegion(self.ra, self.dec, self.radius)
        if not hasattr(self, "chaincat"):
            self.chaincat = make_chaincat(self.chain, self.bands, self.active,
                                          self.reference_coordinates, shapes=self.shape_cols)

    def show_chain(self, source_idx=0, bandlist=None, axes=None, show_shapes=True,
                   span=0.999999426697, post_kwargs=dict(alpha=0.5, color="royalblue"),
                   truth=None, truth_kwargs=dict(linestyle="--", color="tomato")):
        if bandlist is None:
            bandlist = self.bands
        cols = bandlist
        if show_shapes:
            cols = cols + self.shape_cols
        q = 100 * np.array([0.5 - 0.5 * span, 0.5 + 0.5 * span])
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            xx = self.chaincat[source_idx][col]
            lim = np.percentile(xx, list(q))
            if self.bounds is not None:
                lim = self.bounds[source_idx][col]

            ax.plot(xx, **post_kwargs)
            ax.set_ylim(*lim)
            ax.set_ylabel(col)
            if truth is not None:
                ax.axhline(truth[col], **truth_kwargs)
        return ax


def run_lmc(model, q, n_draws, adapt=False, full=False, weight=10,
            warmup=[10], trace=None, z_cov=None, max_treedepth=10,
            discard_tuned_samples=True, progressbar=False,
            random_seed=0xDEADBEEF):
    """Use the littlemcmc barebones NUTS algorithm to sample from the
    posterior.

    Parameters
    ----------
    model : The model object (instance of `forcepho.model.Posterior`)
        This should have the methods `lnprob_and_grad` and `transform()`

    q : ndarray of shape (ndim,)
        The starting position in the constrained parameter space q

    n_draws : int
        Number of posterior samples

    warmup: list of ints
        Number of iterations in each warmup-phase, used to tune the step-size.

    z_cov : optional, ndarray of shape (ndim, ndim)
        The mass matrix, in the unconstrained parameter space `z`

    full : bool, optional (default, False)
        Whether to use the full covariance matrix instead of the diagonal.

    adapt : bool, optional (default, False)
        If true, use a diagonol potential (mass_matrix) that adapts at every
        iteration for the first warmup round. Otherwise the covariance matrix
        will have off diagonal elements and only adapt between warmup rounds.

    Returns
    -------
    chain : ndarray of shape (n_draws, n_dim)
        The posterior samples in the parameter space `q` (i.e. not the sampling
        space `z`)

    step : littlemcmc.nuts.NUTS instance
        The stepper.  See step.potential._cov for the covariance (mass) matrix
        of the step, in `z` units

    stats : list of dicts
        Statistics for each sample
    """
    import warnings
    warnings.filterwarnings('ignore', '.*log1p.*', )

    from littlemcmc.sampling import _sample as sample_one
    from littlemcmc import NUTS

    if model.transform is not None:
        start = model.transform.inverse_transform(q)
    else:
        start = q.copy()
    n_dim = len(start)

    # --- Burn-in windows with step tuning ---
    t = time.time()
    #z_cov, trace = warmup_rounds(warmup, model)

    # --- production run ---
    potential = get_pot(n_dim, adapt=adapt, full=full, weight=weight,
                        init_mean=start, init_cov=z_cov, trace=trace)
    step = NUTS(logp_dlogp_func=model.lnprob_and_grad,
                model_ndim=n_dim, potential=potential,
                max_treedepth=max_treedepth)
    trace, stats = sample_one(logp_dlogp_func=model.lnprob_and_grad,
                              model_ndim=n_dim, start=start, step=step,
                              draws=n_draws, tune=warmup[0],
                              discard_tuned_samples=discard_tuned_samples,
                              progressbar=progressbar,
                              chain=0, random_seed=random_seed)
    tsample = time.time() - t

    if model.transform is not None:
        chain = model.transform.transform(trace.T)
    else:
        chain = trace.T

    result = Result()
    result.ndim = len(q)
    result.starting_position = q.copy()
    result.chain = chain
    result.n_call = model.ncall
    result.wall_time = tsample

    return result, step, stats


def warmup_rounds(warmup, model):
    # for iiter, n_iterations in enumerate(warmup):
    #    if (iiter == 0) & adapt_one:
    #        adapt = True
    #    else:
    #        adapt = False
    #    potential = get_pot(init_cov=z_cov, n_dim=n_dim, trace=trace)
    #    step = NUTS(logp_dlogp_func=model.lnprob_and_grad,
    #                model_ndim=n_dim, potential=potential)
    #    trace, tstats = sample_one(logp_dlogp_func=model.lnprob_and_grad,
    #                               model_ndim=n_dim, start=start, step=step,
    #                               draws=2, tune=n_iterations,
    #                               discard_tuned_samples=False,
    #                               progressbar=progressbar)
    #    start = trace[:, -1]
    #    if adapt:
    #        # Use tuned covariance matrix
    #        z_cov = np.diag(step.potential._var)
    #    else:
    #        # estimate from samples
    #        z_cov = None

    raise(NotImplementedError)


def get_pot(n_dim, init_mean=None, init_cov=None, trace=None,
            regular_variance=1e-3, adapt=True, full=False, weight=10):
    """Generate a full potential (i.e. a mass matrix) either using a supplied
    covariance matrix, a trace of samples, or the identity.

    Returns
    -------
    potential : littlemcmc.quadpotential.QuadPotential instance.
        If `tune` this is an instance of QuadPotentialDiagAdapt with zero mean.
        Otherwise, an instance of QuadPotentialFull, i.e. a static 2-d mass-matrix.
    """
    from littlemcmc import QuadPotentialFull, QuadPotentialFullAdapt, QuadPotentialDiagAdapt
    from scipy.linalg import cholesky

    if init_cov is not None:
        # use_supplied values
        cov = np.array(init_cov)
    elif (trace is None) and (init_cov is None):
        # Default no information
        cov = np.eye(n_dim)
    elif (trace is not None) and (init_cov is None):
        # Estimate from trace
        cov = np.cov(trace, rowvar=True)
        init_mean = np.mean(trace, axis=-1)

    ntimes = 0
    while ntimes < 5:
        try:
            _ = cholesky(cov, lower=True)
            break
        except(np.linalg.LinAlgError):
            cov[np.diag_indices_from(cov)] += regular_variance
            print(ntimes)
            ntimes += 1

    if full:
        init_mass = cov
        potential = QuadPotentialFullAdapt(n_dim, init_mean, init_mass, weight)
    else:
        init_mass = np.diag(cov)
        potential = QuadPotentialDiagAdapt(n_dim, init_mean, init_mass, weight)

    return potential


def run_opt(model, q, jac=True, callback=None, **extras):
    """Simple BFGS optimization using scipy.optimize
    """
    from scipy.optimize import minimize
    #opts = {'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
    #        'disp':True, 'iprint': 1, 'maxcor': 20}
    opts = {}
    opts.update(**extras)

    if model.transform is not None:
        start = model.transform.inverse_transform(q)
    else:
        start = q.copy()


    t0 = time.time()
    if jac:
        scires = minimize(model.nll, start.copy(), jac=jac, method='BFGS',
                          options=opts, bounds=None, callback=callback)
    else:
        scires = minimize(model.nll_nograd, start.copy(), jac=None, method='BFGS',
                          options=opts, bounds=None, callback=callback)

    tsample = time.time() - t0

    result = Result()
    result.ndim = len(q)
    result.starting_position = q.copy()
    result.chain = np.atleast_2d(scires.x)
    if model.transform is not None:
        result.chain = model.transform.transform(result.chain)

    result.lnp = -0.5 * scires.fun
    result.wall_time = tsample

    result.fun = scires.fun
    result.jac = scires.jac
    result.message = scires.message

    return result, scires


def run_opt_bounded(model, q, jac=True, callback=None, **extras):
    """Simple L-BFGS-B optimization using scipy.optimize

    tends not to work very well....
    """
    bounds = extras.pop("bounds", None)
    from scipy.optimize import minimize
    opts = {'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
            'disp':True, 'iprint': 1, 'maxcor': 20}
    opts.update(**extras)

    if model.transform is not None:
        start = model.transform.inverse_transform(q)
        bounds = list([(-200, 200) for p in q])
    else:
        start = q.copy()


    t0 = time.time()
    if jac:
        scires = minimize(model.nll, start.copy(), jac=jac,  method='L-BFGS-B',
                          options=opts, bounds=bounds, callback=callback)
    else:
        scires = minimize(model.nll_nograd, start.copy(), jac=None,  method='L-BFGS-B',
                          options=opts, bounds=bounds, callback=callback)

    tsample = time.time() - t0

    result = Result()
    result.ndim = len(q)
    result.starting_position = q.copy()
    result.chain = np.atleast_2d(scires.x)
    if model.transform is not None:
        result.chain = model.transform.transform(result.chain)

    result.lnp = -0.5 * scires.fun
    result.wall_time = tsample

    result.fun = scires.fun
    result.jac = scires.jac
    result.message = scires.message

    return result, scires


def run_pymc3(model, q, lower=-np.inf, upper=np.inf,
              nwarm=2000, niter=1000):
    """Run a pymc3 fit
    """
    import pymc3 as pm
    import theano.tensor as tt
    import theano
    theano.gof.compilelock.set_lock_status(False)
    from .model import LogLikeWithGrad

    logl = LogLikeWithGrad(model)
    pnames = model.scene.parameter_names
    assert len(upper) == len(lower) == len(pnames)
    t = time.time()
    with pm.Model() as opmodel:
        z0 = [pm.Uniform(p, lower=l, upper=u)
              for p, l, u in zip(pnames, lower, upper)]
        theta = tt.as_tensor_variable(z0)
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=niter, tune=nwarm, cores=1,
                          discard_tuned_samples=True)

    tsample = time.time() - t

    result = Result()
    result.ndim = len(q)
    result.starting_position = q.copy()
    result.chain = np.array([trace.get_values(n) for n in pnames]).T
    result.trace = trace
    result.ncall = model.ncall
    result.wall_time = tsample
    result.lower = lower
    result.upper = upper

    return result


def run_dynesty(model, q, lower=0, upper=1.0, nlive=50):
    """Run a dynesty fit of the model
    """
    import dynesty

    theta_width = (upper - lower)
    ndim = len(lower)

    def prior_transform(unit_coords):
        # now scale and shift
        theta = lower + theta_width * unit_coords
        return theta

    sampler = dynesty.DynamicNestedSampler(model.lnprob, prior_transform, ndim, nlive=nlive,
                                           bound="multi", method="rwalk", bootstrap=0)
    t0 = time.time()
    sampler.run_nested(nlive_init=int(nlive/2), nlive_batch=int(nlive),
                       wt_kwargs={'pfrac': 1.0}, stop_kwargs={"post_thresh": 0.2})
    tsample = time.time() - t0

    dresults = sampler.results

    result = Result()
    result.ndim = ndim
    result.chain = dresults["samples"]
    result.lnp = dresults['logl']
    result.ncall = model.ncall
    result.wall_time = tsample
    result.lower = lower
    result.upper = upper

    return result, dresults


def optimize_one_band(X, w, y, fixedX=0):
    """Linear least-squares to get the ML fluxes and precision matrix for a
    single band.

    Parameters
    ----------
    X : ndarray of shape (nsource, npix_band)
        Normalized model pixel flux for each individual source

    w : ndarray of shape (npix_band,)
        The weights for each pixel (usually 1/sigma)

    fixedX : optional, ndarray of shape (nfixed, npix_band)
        Model pixel fluxes (not normalized) for any fixed sources that are being
        subtracted from the scene.
    """
    Xp = X * w
    yp = (y - fixedX) * w
    ATA = np.dot(Xp, Xp.T)
    Xyp = np.dot(Xp, yp[:, None])
    flux = np.linalg.solve(ATA, Xyp)
    return flux, ATA


def optimize_fluxes(patcher, active, return_all=False):
    """Do a simple wieghted least-squares to get the maximum likelihood fluxes,
    conditional on source shape parameters.

    Returns
    -------
    fluxes : list of ndarrays
        List of ndarrays of shape (n_source,) giving the maximum likelihoood
        fluxes of each source. The list has same length and order as
        `patcher.bandlist`

    precisions : list of ndarrays
        List of ndarrays of shape (n_source, n_source) giving the flux precision
        matrix in each band (i.e. the inverse of the flux covariance matrix)
    """
    from numpy.linalg import LinAlgError

    fluxes, precisions = [], []
    Xes, ys = patcher.design_matrix(active)
    ws = patcher.split_band("ierr")
    for i, (w, X, y) in enumerate(zip(ws, Xes, ys)):
        try:
            flux, precision = optimize_one_band(X, w, y)
            fluxes.append(np.atleast_1d(np.squeeze(flux)))
            precisions.append(precision)
        except(LinAlgError) as e:
            return i, e, Xes, ys, ws
    if return_all:
        return fluxes, precisions, Xes, ys, ws
    else:
        return fluxes, precisions


if __name__ == "__main__":

    from scipy.stats import norm
    from forcepho.model import BoundedTransform, CPUPosterior
    import matplotlib.pyplot as pl
    pl.ion()

    # --- dimension and bounds ---
    ndim = 30
    a = 1.5
    epsilon = 1e-2

    # --- Build the model ---
    def lnlike_gauss(theta, **extras):
        lnp = -0.5 * np.dot(theta, theta)
        lnp_grad = -theta
        return lnp, lnp_grad

    if a is not None:
        lower = np.zeros(ndim) - a
        upper = np.zeros(ndim) + a
        transform = BoundedTransform(lower, upper)
    else:
        transform = None
        upper = np.inf
        lower = -np.inf
    model = CPUPosterior(lnlike=lnlike_gauss, transform=transform)
    q = np.clip(np.random.normal(0, 1, size=(ndim,)), lower + epsilon, upper - epsilon)

    # --- Run the sampler ---
    chain, step, stats = run_lmc(model, q, n_draws=1000,
                                 warmup=[256, 512, 1024, 1024, 1024, 2048])

    if a is not None:
        from scipy.special import erf
        A = erf(a / np.sqrt(2))
    else:
        A = 1

    fig, ax = pl.subplots()
    hh = [ax.hist(chain[:, i], histtype="step", bins=np.linspace(-4, 4, 100), alpha=0.5, density=True)
          for i in range(ndim)]
    xx = np.linspace(-10, 10, 1000)
    ax.plot(xx, norm.pdf(xx) / A)
    pl.show()