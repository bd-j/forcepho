#!/usr/bin/env python

"""
A script to read in many XDF patches and launch a sampler on each patch

On Ascent with CUDA MPS, one could run this with 16 processes on 7 cores using 1 GPU with:
$ jsrun -n1 -a16 -c7 -g1 ./run_patch_gpu_test.py
"""

import sys, os
from time import time
from os.path import join as pjoin
import numpy as np
from scipy.optimize import minimize
try:
    import cPickle as pickle
except(ImportError):
    import pickle

from forcepho.patch import Patch
from forcepho.proposal import Proposer
from forcepho.kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS
from forcepho.posterior import LogLikeWithGrad
from forcepho.fitting import Result

from patch_conversion import patch_conversion, zerocoords

import theano
import pymc3 as pm
import theano.tensor as tt
theano.gof.compilelock.set_lock_status(False)
# be quiet
import logging
logger = logging.getLogger("pymc3")
logger.propagate = False
logger.setLevel(logging.ERROR)

try:
    import pycuda.autoinit
    scratch_dir = pjoin('/gpfs/wolf/gen126/scratch', os.environ['USER'], 'residual_images')
    os.makedirs(scratch_dir, exist_ok=True)
except:
    HAS_GPU = False
_print = print
print = lambda *args,**kwargs: _print(*args,**kwargs, file=sys.stderr, flush=True)


def save_results(result, rname):
   if rname is not None:
        with open("{}.pkl".format(rname), "wb") as f:
            pickle.dump(result, f)


class GPUPosterior:

    def __init__(self, proposer, scene):
        self.proposer = proposer
        self.scene = scene
        self.ncall = 0
        self._z = -99

    def evaluate(self, z):
        """
        :param z: 
            The untransformed (sampling) parameters which have a prior
            distribution attached.

        Theta are the transformed forcepho native parameters.  In the default
        case these are these are the same as thetaprime, i.e. the transformation
        is the identity.
        """
        self.scene.set_all_parameters(z)
        proposal = self.scene.get_proposal()
        
        # send to gpu and collect result   
        ret = self.proposer.evaluate_proposal(proposal)
        if len(ret) == 3:
            chi2, chi2_derivs, _ = ret
        else:
            chi2, chi2_derivs = ret

        # turn into log-like and accumulate grads correctly
        ll = -2 * chi2
        ll_grad = -2 * self.stack_grad(chi2_derivs)
     
        self.ncall += 1
        self._lnp = ll 
        self._lnp_grad = ll_grad
        self._z = z

    def stack_grad(self, chi2_derivs):
        """The chi2_derivs is returned as an array of shape NBAND, NACTIVE, NPARAMS.
        
        final output should be [flux11, flux12, ..., flux1Nb, ra1, dec1, ..., rh1, 
                                flux21, flux22, ..., flux2Nb, ra2, dec2, ..., rh2]
        """
        nsources = self.proposer.patch.n_sources
        nbands = self.proposer.patch.n_bands #len(chi2_derivs)
        grads = np.zeros([nbands, nsources, nbands + (NPARAMS-1)])
        for band, derivs in enumerate(chi2_derivs):            
            # shape params
            grads[:, nbands:] += derivs[:, 1:]
            # flux params
            grads[:, band] += derivs[:, 0]
            
        return grads.reshape(-1)

    def lnprob(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp

    def lnprob_grad(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return self._lnp_grad
    
    def nll(self, z):
        if np.any(z != self._z):
            self.evaluate(z)
        return -self._lnp, -self._lnp_grad
    
    def residuals(self, z):
        assert self.proposer.patch.return_residuals
        self.scene.set_all_parameters(z)
        proposal = self.scene.get_proposal()
        ret = self.proposer.evaluate_proposal(proposal)
        chi2, chi2_derivs, self.residuals = ret
        return self.residuals



def prior_bounds(scene, npix=3, flux_factor=20):
    """ Priors and scale guesses:
    
    :param scene:
        A Scene object.  Each source must have the `stamp_cds` attribute.
        
    :param npix: (optional, default: 3)
        Number of pixels to adopt as the positional prior
    """
    dpix_dsky = scene.sources[0].stamp_cds[0]
    source = scene.sources[0]
    plate_scale = np.abs(np.linalg.eigvals(np.linalg.inv(dpix_dsky)))
    rh_range = np.array(scene.sources[0].rh_range)
    sersic_range = np.array(scene.sources[0].sersic_range)

    lower = [s.nband * [0.] +
             [s.ra - npix * plate_scale[0], s.dec - npix * plate_scale[1],
              0.3, -np.pi/1.5, sersic_range[0], rh_range[0]]
             for s in scene.sources]
    upper = [(np.array(s.flux) * flux_factor).tolist() +
             [s.ra + npix * plate_scale[0], s.dec + npix * plate_scale[1],
              1.0, np.pi/1.5, sersic_range[-1], rh_range[-1]]
             for s in scene.sources]
    lower = np.concatenate(lower)
    upper = np.concatenate(upper)

    return lower, upper


path_to_data = "/gpfs/wolf/gen126/proj-shared/jades/udf/data/"
path_to_results = "/gpfs/wolf/gen126/proj-shared/jades/udf/results/"
#patch_name = pjoin(path_to_data, "test_patch_mini.h5")  # "test_patch_large.h5" or test_patch.h5 or "test_patch_mini.h5"
splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
psfpath = path_to_data

def run_patch(patchname, splinedata=splinedata, psfpath=psfpath, 
              nwarm=200, niter=500, sample=True):
    """
    This runs in a single CPU process.  It dispatches the 'patch data' to the
    device and then runs a pymc3 HMC sampler.  Each likelihood call within the
    HMC sampler copies the proposed parameter position to the device, runs the
    kernel, and then copies back the result, returning the summed ln-like and
    the gradients thereof to the sampler.

    :param patchname: 
        Full path to the patchdata hdf5 file.
    """

    resultname = os.path.basename(patchname).replace("h5", "pkl")
    resultname = pjoin(path_to_results, resultname)
    try:
        r = pool.rank
    except:
        r = 0
    print("Rank {} writing to {}".format(r, resultname))

    # --- Prepare Patch Data ---
    stamps, miniscene = patch_conversion(patchname, splinedata, psfpath, nradii=9)
    pra = np.median([s.ra for s in miniscene.sources])
    pdec = np.median([s.dec for s in miniscene.sources])
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))
    patch = Patch(stamps=stamps, miniscene=miniscene, return_residual=True)
    p0 = miniscene.get_all_source_params().copy()

    # --- Copy patch data to device ---
    gpu_patch = patch.send_to_gpu()
    gpu_proposer = Proposer(patch)

    # --- Instantiate the ln-likelihood object ---
    # This object splits the lnlike_function into two, since that computes 
    # both lnp and lnp_grad, and we need to wrap them in separate theano ops.
    model = GPUPosterior(gpu_proposer, miniscene)

    # --- Subtract off the fixed sources ---
    # TODO

    if sample:
        # -- Launch HMC ---
        # wrap the loglike and grad in theano tensor ops
        model.proposer.patch.return_residuals = False
        logl = LogLikeWithGrad(model)
        # Get upper and lower bounds for variables
        lower, upper = prior_bounds(scene)
        pnames = scene.parameter_names
        # The pm.sample() method below will draw an initial theta, 
        # then call logl.perform and logl.grad multiple times
        # in a loop with different theta values.
        t = time()
        with pm.Model() as opmodel:
            # set priors for each element of theta
            z0 = [pm.Uniform(p, lower=l, upper=u) 
                  for p, l, u in zip(pnames, lower, upper)]
            theta = tt.as_tensor_variable(z0)
            # instantiate target density and start sampling.
            pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
            trace = pm.sample(draws=niter, tune=nwarm, progressbar=False,
                              cores=1, discard_tuned_samples=True)

        ts = time() - t
        # yuck.
        chain = np.array([trace.get_values(n) for n in pnames]).T
        
        result = Result()
        result.ndim = len(p0)
        result.pinitial = p0.copy()
        result.chain = chain
        result.trace = trace
        result.ncall = model.ncall
        result.wall_time = ts
        result.scene = miniscene
        result.lower = lower
        result.upper = upper
        result.patchname = patchname
        result.sky_reference = (pra, pdec)
        
        #last = chain[:, -1]
        #model.proposer.patch.return_residuals = True
        #result.residuals = model.residuals(last)

        save_results(result, resultname)

    else:
        # --- Launch an optimization ---
        opts = {'ftol': 1e-6, 'gtol': 1e-6, 'factr': 10.,
                'disp':False, 'iprint': 1, 'maxcor': 20}
        theta0 = miniscene.get_all_source_params().copy()
        t = time()
        scires = minimize(model.nll, theta0, jac=True,  method='BFGS',
                          options=opts, bounds=None)
        ts = time() - t
        chain = scires       
    
    return chain, (r, model.ncall, ts)

# Create an MPI process pool.  Each worker process will sit here waiting for input from master
# after it is done it will get the next value from master.
try:
    from emcee.utils import MPIPool
    pool = MPIPool(debug=False, loadbalance=False)
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
except(ImportError, ValueError):
    pool = None
    print('Not using MPI')


def halt(message):
    """Exit, closing pool safely.
    """
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":
    patches = [23, 530]
    allpatches = [pjoin(path_to_data, "patches", "patch_udf_withcat_{}.h5".format(pid)) 
                  for pid in patches]
    
    try:
        M = pool.map
    except:
        M = map

    t = time()
    allchains = M(run_patch, allpatches)
    twall = time() - t

    halt("finished all patches in {}s".format(twall))
