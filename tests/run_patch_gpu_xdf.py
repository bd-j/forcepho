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
import h5py

from forcepho.patch import StaticPatch
from forcepho.proposal import Proposer
from forcepho.kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS
from forcepho.model import LogLikeWithGrad
from forcepho.fitting import Result
from forcepho.model import GPUPosterior

from patch_conversion import patch_conversion, zerocoords, set_inactive

import theano
import pymc3 as pm
import theano.tensor as tt
theano.gof.compilelock.set_lock_status(False)
# be quiet
import logging
logger = logging.getLogger("pymc3")
logger.propagate = False
logger.setLevel(logging.ERROR)


import pycuda.autoinit
scratch_dir = pjoin('/gpfs/wolf/gen126/scratch',
                    os.environ['USER'], 'residual_images')
os.makedirs(scratch_dir, exist_ok=True)

_print = print
print = lambda *args, **kwargs: _print(*args, **kwargs, file=sys.stderr, flush=True)


def prior_bounds(scene, npix=4, flux_factor=4):
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


def run_patch(patchname, maxactive=3,
              splinedata="", psfpath="",
              resultspath="",
              init_cov=None, start=None,
              nwarm=250, niter=200, ntime=10,
              runtype="sample", verbose=True,
              rank=0, scatter_fluxes=False, tag=""):
    """
    This runs in a single CPU process.  It dispatches the 'patch data' to the
    device and then runs a pymc3 HMC sampler.  Each likelihood call within the
    HMC sampler copies the proposed parameter position to the device, runs the
    kernel, and then copies back the result, returning the summed ln-like and
    the gradients thereof to the sampler.

    :param patchname: 
        Full path to the patchdata hdf5 file.
    """
    """
    This runs in a single CPU process.  It dispatches the 'patch data' to the
    device and then runs a pymc3 HMC sampler.  Each likelihood call within the
    HMC sampler copies the proposed parameter position to the device, runs the
    kernel, and then copies back the result, returning the summed ln-like and
    the gradients thereof to the sampler.

    :param patchname: 
        Full path to the patchdata hdf5 file.
    """

    print(patchname)

    #if tag is not None:
    #    tail = "_{}_result"

    resultname = tag + "-" + os.path.basename(patchname).replace(".h5", "_result")
    resultname = pjoin(path_to_results, resultname)

    print("Rank {} writing to {}".format(rank, resultname))

    # --- Prepare Patch Data ---
    use_bands = slice(None)
    stamps, scene = patch_conversion(patchname, splinedata, psfpath,
                                     nradii=9, use_bands=use_bands)
    miniscene = set_inactive(scene, [stamps[0], stamps[-1]], nmax=maxactive)
    pra = np.median([s.ra for s in miniscene.sources])
    pdec = np.median([s.dec for s in miniscene.sources])
    zerocoords(stamps, miniscene, sky_zero=np.array([pra, pdec]))

    if scatter_fluxes:
        for s in miniscene.sources:
            s.flux = s.flux + np.arange(0, len(s.filternames)) * 0.1

    patch = Patch(stamps=stamps, miniscene=miniscene, return_residual=True)
    p0 = miniscene.get_all_source_params().copy()

    # --- Copy patch data to device ---
    gpu_patch = patch.send_to_gpu()
    gpu_proposer = Proposer(patch)

    # --- Instantiate the ln-likelihood object ---
    # This object splits the lnlike_function into two, since that computes
    # both lnp and lnp_grad, and we need to wrap them in separate theano ops.
    model = GPUPosterior(gpu_proposer, miniscene, name=patchname, verbose=verbose)

    # --- Subtract off the fixed sources ---
    # TODO

    if runtype == "sample":
        # -- Launch HMC ---
        # wrap the loglike and grad in theano tensor ops
        model.proposer.patch.return_residuals = False
        logl = LogLikeWithGrad(model)
        # Get upper and lower bounds for variables
        lower, upper = prior_bounds(miniscene)
        print(lower.dtype, upper.dtype)
        pnames = miniscene.parameter_names
        start = dict(zip(pnames, p0))
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
        result.nactive = miniscene.nactive
        result.nbands = patch.n_bands
        result.nexp = patch.n_exp
        result.pinitial = p0.copy()
        result.chain = chain
        result.ncall = np.copy(model.ncall)
        result.wall_time = ts
        #result.scene = miniscene
        result.lower = lower
        result.upper = upper
        result.patchname = patchname
        result.sky_reference = (pra, pdec)
        result.parameter_names = pnames

        #last = chain[:, -1]
        #model.proposer.patch.return_residuals = True
        #result.residuals = model.residuals(last)

        save_results(result, resultname)

    elif runtype == "optimize":
        # --- Launch an optimization ---
        opts = {'ftol': 1e-6, 'gtol': 1e-6, 'factr': 10.,
                'disp':False, 'iprint': 1, 'maxcor': 20}
        theta0 = p0
        t = time()
        scires = minimize(model.nll, theta0, jac=True,  method='BFGS',
                          options=opts, bounds=None)
        ts = time() - t
        chain = scires

    elif runtype == "timing":
        # --- Time a single call ---
        model.proposer.patch.return_residuals = False
        t = time()
        for i in range(ntime):
            model.evaluate(p0)
            print(model._lnp)
            print(model._lnp_grad)
        ts = time() - t
        chain = [model._lnp, model._lnp_grad]
        print("took {}s for a single call".format(ts / ntime))

    return chain, (rank, model.ncall, ts)


def save_results(result, rname):
   if rname is not None:
        with h5py.File("{}.h5".format(rname), "w") as f:
            f.create_dataset("chain", data=result.chain)
            f.attrs["ncall"] = result.ncall
            f.attrs["wall_time"] = result.wall_time
            f.attrs["patchname"] = result.patchname
            f.attrs["sky_reference"] = result.sky_reference
            f.attrs["nactive"] = result.nactive
            f.attrs["nbands"] = result.nbands
            f.attrs["parameters"] = result.parameter_names.astype("S")
            f.attrs["lower_bounds"] = result.lower
            f.attrs["upper_bounds"] = result.upper


def distribute_patches(allpatches, mpi_barrier=False, run_kwargs={}):
    try:
        from mpi4py import MPI
        have_mpi = True
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except:
        have_mpi = False
        rank = 0
        size = 1
        assert not mpi_barrier

    import time
    if rank == 0:
        print("{} ranks handling {} patches".format(size, len(allpatches)))
        print(allpatches)

    for i in range(rank, len(allpatches), size):
        patch = allpatches[i]
        time.sleep(rank * 2)
        out = run_patch(patch, rank=rank, **run_kwargs)



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

    parser = ArgumentParser()
    parser.add_argument("--xdf_path", type=str,
                        default="/n/scratchlfs/eisenstein_lab/bdjohnson/xdf")
    parser.add_argument("--patch_dir", type=str,
                        default="")
    parser.add_argument("--patch_number", type=int, nargs="*",
                        default=[90, 91])
    parser.add_argument("--patch_name", type=str,
                        default="patch_udf_withcat_{}.h5")
    args = parser.parse_args()

    xdf_path = args.xdf_path
    patches = args.patch_number
    patchdir = args.patch_dir
    if patchdir == "":
        patchdir = pjoin(xdf_path, "data", "patches_with_cat")
    patchname = args.patch_name

    path_to_data = pjoin(xdf_path, "data")
    results_path = pjoin(xdf_path, "results")
    splinedata = pjoin(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = pjoin(path_to_data, "psfs", "mixtures")
    #scratch_dir = pjoin(xdf_path, "cannon", "residual_images")
    #os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    #patches = [100, 183, 274, 382, 653]
    #patches2 = [90, 91, 157, 159, 160, 172, 212]
    #patches3 = [90, 91]
    allpatches = [pjoin(patchdir, patchname.format(pid))
                  for pid in patches]

    run_kwargs = {"maxactive": 15,
                  "tag": "max15",
                  "nwarm": 100,
                  "niter": 100,
                  "ntune": 200,
                  "runtype": "sample",
                  "scatter_fluxes": False,
                  "psfpath": psfpath,
                  "splinedata": splinedata,
                  "resultspath": results_path
                  }

    t = time()
    distribute_patches(allpatches, run_kwargs=run_kwargs)
    twall = time() - t

    halt("finished all patches in {}s".format(twall))
