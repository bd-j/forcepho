#!/usr/bin/env python

import os
import sys
sys.path.append('..')
from os.path import join as pjoin

import numpy as np

from forcepho.patch import Patch
from forcepho.proposal import Proposer

from patch_conversion import patch_conversion

import pycuda.autoinit

scratch_dir = pjoin('/gpfs/wolf/gen126/scratch', os.environ['USER'], 'residual_images')
os.makedirs(scratch_dir, exist_ok=True)


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


if __name__ == '__main__':
    path_to_data = '/gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data'
    patch_name = os.path.join(path_to_data, "test_patch_large.h5")
    splinedata = os.path.join(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = path_to_data
    nradii = 9

    list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)

    patch = Patch(stamps=list_of_stamps, miniscene=mini_scene, return_residual=True)
    gpu_patch = patch.send_to_gpu()
    #patch.test_struct_transfer(gpu_patch)

    proposal = mini_scene.get_proposal()
    proposer = Proposer(patch)

    import time

    niter = 3
    start = time.time()

    for i in range(niter):
        chi2, chi2_derivs, residuals = proposer.evaluate_proposal(proposal)

    end = time.time()

    print(f'Elapsed: {end-start:.3g}s for {niter} proposals')

    plot_residuals(patch, residuals)

    #print(residuals[0][110,5])

    #datas = proposer.unpack_residuals(patch.data)
    #plot_residuals(patch, datas)
