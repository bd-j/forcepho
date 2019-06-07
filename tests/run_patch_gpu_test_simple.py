#!/usr/bin/env python

'''

A simple test sccript to set up one patch and launch many proposals on it.

On Ascent with CUDA MPS, one could run this with 16 processes on 7 cores using 1 GPU with:

$ jsrun -n1 -a16 -c7 -g1 ./run_patch_gpu_test_simple.py

'''

import os
import sys
from os.path import join as pjoin
import time

sys.path.append('..')

import numpy as np

from forcepho.patch import Patch
from forcepho.proposal import Proposer

from patch_conversion import patch_conversion

import pycuda.autoinit

scratch_dir = pjoin('/gpfs/wolf/gen126/scratch', os.environ['USER'], 'residual_images')
os.makedirs(scratch_dir, exist_ok=True)

_print = print
print = lambda *args,**kwargs: _print(*args,**kwargs, file=sys.stderr, flush=True)


def plot_residuals(patch, residuals, vmin=None, vmax=None):
    print('Plotting residuals...')
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # find global residual min and max for colorbar purposes
    if not vmin:
        vmin = np.min([np.min(r) for r in residuals])
    if not vmax:
        vmax = np.max([np.max(r) for r in residuals])

    n_exp_max = np.max(patch.band_N)

    figsize = np.array([n_exp_max, patch.n_bands])
    fig, axes = plt.subplots(patch.n_bands, n_exp_max, dpi=72, figsize=figsize*4, squeeze=False)
    for b in range(patch.n_bands):
        axrow = axes[b]
        for e in range(patch.band_N[b]):
            eindex = patch.band_start[b] + e
            ax = axrow[e]
            ax.set_aspect('equal')
            im = ax.imshow(residuals[eindex].T, vmin=vmin, vmax=vmax, origin='lower')

    fig.subplots_adjust(left=0.02, wspace=0.01, hspace=0.01)
    cbar_ax = fig.add_axes([0.005, 0.15, 0.005, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    #fig.tight_layout()

    savefn = pjoin(scratch_dir, 'residuals_grid.png')
    fig.savefig(savefn)
    print(f'Plotted to {savefn}')


def time_proposals(n_repeat=100, mpi_barrier=True):
    '''
    Launch a proposal n_repeat times.

    mpi_barrier will wait until all processes are ready to run
    proposals, in hope of getting everything to hit the GPU
    at the same time.
    '''

    try:
        raise
        from mpi4py import MPI
        have_mpi = True
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except:
        have_mpi = False
        assert not mpi_barrier

    if mpi_barrier:
        # use subcommunicator per-GPU?
        comm.barrier()
        if rank == 0:
            print(f'Passed MPI barrier with {size} ranks')

    start = time.time()

    for i in range(n_repeat):
        ret = proposer.evaluate_proposal(proposal)

    # barrier at the end?

    end = time.time()

    print(f'Elapsed: {end-start:.3g}s for {n_repeat} proposals')


if __name__ == '__main__':
    path_to_data = '/gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data'
    patch_name = os.path.join(path_to_data, "test_patch_mini.h5")  # "test_patch_large.h5" or test_patch.h5 or "test_patch_mini.h5"
    splinedata = os.path.join(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = path_to_data
    nradii = 9

    list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)

    patch = Patch(stamps=list_of_stamps, miniscene=mini_scene, return_residual=True)
    gpu_patch = patch.send_to_gpu()
    #patch.test_struct_transfer(gpu_patch)

    proposal = mini_scene.get_proposal()
    proposer = Proposer(patch)

    # One for burn-in before timing
    ret = proposer.evaluate_proposal(proposal)

    if len(ret) == 3:
        chi2, chi2_derivs, residuals = ret

        plot_residuals(patch, residuals)

    else:
        chi2, chi2_derivs = ret

    print(chi2, chi2_derivs)

    #time_proposals()

    #print(residuals[0][110,5])

    #datas = proposer.unpack_residuals(patch.data)
    #plot_residuals(patch, datas)
