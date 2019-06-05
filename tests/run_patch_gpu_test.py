import sys, os, time
import numpy as np
from scipy.optimize import minimize

from frocepho.posterior import Posterior
from forcepho.patch import Patch

from patch_conversion import patch_conversion

try:
    import theano
    import pymc3 as pm
    import theano.tensor as tt
    theano.gof.compilelock.set_lock_status(False)
    # quiet
    import logging
    logger = logging.getLogger("pymc3")
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    HAS_SAMPLER = True

    from lnlike import LogLikeWithGrad

except(ImportError):
    HAS_SAMPLER = False

try:
    import pycuda.driver as cuda
    from pycuda import gpuarray, compiler, autoinit
    HAS_GPU = True
    CACHE_DIR = os.env["PYCUDA_CACHE"]
except:
    HAS_GPU = False

#HAS_SAMPLER = False
#HAS_GPU = False

if HAS_GPU:
    grid_dim = (1,1)
    block_dim = (1024, 1, 1)
    # load the kernel
    # This C source code can be read from a file
    with open(kernel_file, "r") as f:
        kernel_string = f.readlines()
    mod = compiler.SourceModule(kernel_string, 
                                cache_dir=CACHE_DIR)

    lnlike_kernel = mod.get_function("EvaluateProposal")


def gpu_lnlike_multi(Theta, scene, patchdata_dev):
    # copy theta to device
    scene.set_all_parameters(Theta)
    proposal = scene.get_proposal()
    proposal_dev = send_proposal(proposal)

    # --- make room for output ---
    grad_dev = gpuarray.empty(Theta.shape, np.float32)
    
    # execute kernel
    lnlike_gauss(proposal_dev_dev, patchdata_dev, lnp, grad_dev, ndim,
                 block=block_dim, grid=grid_dim)
    
    # read result
    lnlike = lnp
    grad = grad_dev.get()
    # presumably we could do this accumulation on the gpu
    # for each stamp

    return lnlike.astype(np.float32), grad.astype(np.float32)

splinedataname = "//.h5"
psfpath = "//"

def run_patch(patchname, splinedataname=splinedatafile, psfpath=psfpath, 
              nwarm=50, niter=500, gpu=HAS_GPU, sample=HAS_SAMPLER):
    """
    This runs in a single CPU process.  It dispatches the 'patch data' (cov) 
    to the device and then runs a pymc3 HMC sampler.  Each likelihood call 
    within the HMC sampler copies the proposed parameter position to the device,
    runs the kernel, and then copies back the result, returning the summed ln-like 
    and the gradients thereof to the sampler.
    
    :param patchinfo:
        A list of diagonal elements of a covariance matrix
    """
    
    # --- Prepare Patch Data ---
    stamps, miniscene = patch_conversion(patchname, splinedataname, psfpath, n_psf=9)
    patch = Patch(stamps, miniscene)
    
    # --- Copy patch data to device ---
    if gpu:
        print("using GPU")
        patchdata_dev = patch.send_to_gpu()
        
    
    # --- Instantiate the ln-likelihood object ---
    # This object splits the lnlike_function into two, since that computes 
    # both lnp and lnp_grad, and we need to wrap them in separate theano ops.
    if gpu:
        LL = Posterior(gpu_lnlike_multi, miniscene, patchdata_dev)
    else:
        LL = Posterior(lnlike_multi, miniscene, stamps)
    
    if sample:
        # -- Launch HMC ---
        # wrap the loglike and grad in theano tensor ops
        logl = LogLikeWithGrad(LL)
        # The pm.sample() method below will draw an initial theta, 
        # then call logl.perform and logl.grad multiple times
        # in a loop with different theta values.
        t = time.time()
        with pm.Model() as opmodel:
            # set priors for each element of theta
            z0 = [pm.Uniform(p, lower=-50, upper=50)
                for p in pnames]
            theta = tt.as_tensor_variable(z0)
            # instantiate target density and start sampling.
            pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
            trace = pm.sample(draws=niter, tune=nwarm,
                            cores=1, discard_tuned_samples=True,
                            progressbar=False)

        ts = time.time() - t
        # yuck.
        chain = np.array([trace.get_values(n) for n in pnames]).T

    else:
        # --- Launch an optimization ---
        if gpu:
            def nll(theta):
                lnp, grad = gpu_lnlike_function(theta, cov_dev)
                return -lnp, -grad
        else:
            def nll(theta):
                lnp, grad = cpu_lnlike_function(theta, cov)
                return -lnp, -grad

        opts = {'ftol': 1e-6, 'gtol': 1e-6, 'factr': 10.,
                'disp':False, 'iprint': 1, 'maxcor': 20}
        theta0 = np.ones(ndim)
        t = time.time()
        scires = minimize(nll, theta0, jac=True,  method='BFGS',
                        options=opts, bounds=None)
        ts = time.time() - t
        chain = scires

    try:
        r = pool.rank
    except:
        r = 0
    
    return chain, (r, LL.ncall, ts)
    
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
    
    allpatches = [patchname1, patchname2]
    
    try:
        M = pool.map
    except:
        M = map

    t = time.time()
    allchains = M(run_patch, allpatches)
    twall = time.time() - t
    
    ncall = 0
    tsonly = 0.
    
    print(twall)

    if HAS_SAMPLER:
        for p, (c, info) in zip(allpatches, allchains):
            print(p)
            print(info)
            ncall += info[1]
            tsonly += info[2]
            cov = (np.linspace(p[0], p[0] + 3, p[1]))**2
            sd = c.std(axis=0)
            print(np.sqrt(cov))
            print(sd)
            # this is not quite right.  Also doesn't account for 
            # effective sample size, which is smaller than the number of samples
            frac_sigma_sd = 1 / np.sqrt(2 * c.shape[0] - 2)
            # this will fail very occassionally
            assert np.all(np.abs(np.sqrt(cov) / sd - 1) < (5 * frac_sigma_sd))
            print("----")
        
        print(tsonly, twall, ncall, tsonly/ncall)
    else:
        for p, (c, info) in zip(allpatches, allchains):
            print(p)
            print(info)
            print(c.nfev, c.message, len(c.x))
            assert np.allclose(c.x, 0.0, atol=1e-4)
            print("----")
    
    halt("finished all patches")
