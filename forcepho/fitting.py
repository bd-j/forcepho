import sys, os, time
from functools import partial as argfix
import numpy as np

from .likelihood import lnlike_multi, negative_lnlike_multi
from .posterior import Posterior
try:
    from .posterior import  LogLikeWithGrad
except(ImportError):
    HAS_PYMC3 = False


__all__ = ["Result", "run_pymc3", "run_opt",
           "run_hemcee", "run_dynesty", "run_hmc"]


class Result(object):
    """A simple namespace for storing information about a run.
    """
    
    def __init__(self):
        self.offsets = None


def priors(scene, stamps, npix=2.0):
    # --------------------------------
    # --- Priors ---
    plate_scale = np.abs(np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky)))
    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)

    lower = [s.nband * [0.] +
             [s.ra - npix * plate_scale[0], s.dec - npix * plate_scale[1],
              0.3, -np.pi/1.5, sersic_range[0], rh_range[0]]
             for s in scene.sources]
    upper = [(np.array(s.flux) * 10).tolist() +
             [s.ra + npix * plate_scale[0], s.dec + npix * plate_scale[1],
              1.0, np.pi/1.5, sersic_range[-1], rh_range[-1]]
             for s in scene.sources]
    upper = np.concatenate(upper)
    lower = np.concatenate(lower)
    fluxes = np.array([s.flux for s in scene.sources])
    scales = np.concatenate([f.tolist() + [plate_scale[0], plate_scale[1], 0.1, 0.1, 0.1, 0.01]
                             for f in fluxes])

    return scales, lower, upper


def run_pymc3(p0, scene, plans, lower=-np.inf, upper=np.inf,
              priors=None, nwarm=2000, niter=1000):

    import pymc3 as pm
    import theano.tensor as tt

    model = Posterior(scene, plans)
    logl = LogLikeWithGrad(model)
    pnames = scene.parameter_names
    assert len(upper) == len(lower) ==len(pnames)
    t = time.time()
    with pm.Model() as opmodel:
        if priors is None:
            z0 = [pm.Uniform(p, lower=l, upper=u)
                for p, l, u in zip(pnames, lower, upper)]
        else:
            z0 = priors
        theta = tt.as_tensor_variable(z0)
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        trace = pm.sample(draws=niter, tune=nwarm, cores=1, discard_tuned_samples=True)

    tsample = time.time() - t
    
    result = Result()
    result.ndim = len(p0)
    result.pinitial = p0.copy()
    result.chain = np.array([trace.get_values(n) for n in pnames]).T
    result.trace = trace
    #result.lnp = sampler.lnp.copy()
    result.ncall = model.ncall
    result.wall_time = tsample
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper

    return result


def run_hemcee(p0, scene, plans, scales=1.0, nwarm=2000, niter=1000):
    
    # --- hemcee ---

    from hemcee import NoUTurnSampler
    from hemcee.metric import DiagonalMetric
    metric = DiagonalMetric(scales**2)
    model = Posterior(scene, plans, upper=np.inf, lower=-np.inf)
    sampler = NoUTurnSampler(model.lnprob, model.lnprob_grad, metric=metric)

    result = Result()
    result.ndim = len(p0)
    result.pinitial = p0.copy()
    
    t = time.time()
    pos, lnp0 = sampler.run_warmup(p0, nwarm)
    twarm = time.time() - t
    ncwarm = np.copy(model.ncall)
    model.ncall = 0
    t = time.time()
    chain, lnp = sampler.run_mcmc(pos, niter)
    tsample = time.time() - t
    ncsample = np.copy(model.ncall)

    result.chain = chain
    result.lnp = lnp
    result.ncall = (ncwarm, ncsample)
    result.wall_time = (twarm, tsample)
    result.plans = plans
    result.scene = scene
    result.metric_variance = np.copy(metric.variance)
    result.step_size = sampler.step_size.get_step_size()

    return result


def run_dynesty(scene, plans, lower=0, upper=1.0, nlive=50):

    # --- nested ---
    lnlike = argfix(lnlike_multi, scene=scene, plans=plans, grad=False)
    theta_width = (upper - lower)
    ndim = len(lower)
    
    def prior_transform(unit_coords):
        # now scale and shift
        theta = lower + theta_width * unit_coords
        return theta

    import dynesty
    sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, ndim, nlive=nlive,
                                           bound="multi", method="slice", bootstrap=0)
    t0 = time.time()
    sampler.run_nested(nlive_init=int(nlive/2), nlive_batch=int(nlive),
                       wt_kwargs={'pfrac': 1.0}, stop_kwargs={"post_thresh":0.2})
    tsample = time.time() - t0

    dresults = sampler.results
    
    result = Result()
    result.ndim = ndim
    result.chain = dresults["samples"]
    result.lnp = dresults['logl']
    #result.ncall = nsample
    result.wall_time = tsample
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper

    return result, dresults


def run_opt(p0, scene, plans, jac=True, **extras):

    nll = argfix(negative_lnlike_multi, scene=scene, plans=plans, grad=jac)

    from scipy.optimize import minimize
    opts = {'ftol': 1e-5, 'gtol': 1e-5, 'factr': 10.,
            'disp':True, 'iprint': 1, 'maxcor': 20}
    def callback(x):
        print(x, nll(x))

    t0 = time.time()
    scires = minimize(nll, p0.copy(), jac=jac,  method='BFGS',
                      options=opts, bounds=None, callback=callback)
    tsample = time.time() - t0
    
    result = Result()
    result.ndim = len(p0)
    result.chain = np.atleast_2d(scires.x)
    result.lnp = -0.5 * scires.fun
    result.wall_time = tsample
    result.plans = plans
    result.scene = scene

    return result

    
def run_hmc(p0, scene, plans, scales=1.0, lower=-np.inf, upper=np.inf,
            nwarm=0, niter=500, length=20):

    # -- hmc ---
    from hmc import BasicHMC
    model = Posterior(scene, plans, upper=upper, lower=lower, verbose=True)
    sampler = BasicHMC(model, verbose=False)
    sampler.ndim = len(p0)
    sampler.set_mass_matrix(1/scales**2)

    result = Result()
    result.pinitial = p0.copy()
    
    eps = sampler.find_reasonable_stepsize(p0*1.0)
    use_eps = result.step_size * 2
    result.step_size = np.copy(use_eps)
    result.metric = scales**2

    if nwarm > 0:
        pos, prob, grad = sampler.sample(p0, iterations=nwarm, mass_matrix=1/scales**2,
                                        epsilon=use_eps, length=length, sigma_length=int(length/4),
                                        store_trajectories=True)
        use_eps = sampler.find_reasonable_stepsize(pos)
        result.step_size = np.copy(use_eps)
        ncwarm = np.copy(model.ncall)
        model.ncall = 0

    pos, prob, grad = sampler.sample(pos, iterations=niter, mass_matrix=1/scales**2,
                                     epsilon=use_eps, length=length, sigma_length=int(length/4),
                                     store_trajectories=True)

    result.ndim = len(p0)
    result.chain = sampler.chain.copy()
    result.lnp = sampler.lnp.copy()
    result.plans = plans
    result.scene = scene
    result.lower = lower
    result.upper = upper
    result.trajectories = sampler.trajectories
    result.accepted = sampler.accepted
    result.ncall = (ncwarm, np.copy(model.ncall))

    return result
