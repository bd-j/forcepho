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

if __name__ == '__main__':
    path_to_data = '/gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data'
    patch_name = os.path.join(path_to_data, "test_patch.h5")
    splinedata = os.path.join(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = path_to_data
    nradii = 9

    list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)

    p = Patch(stamps=list_of_stamps, miniscene=mini_scene, return_residual=True)
    gpu_patch = p.send_to_gpu()
    #p.test_struct_transfer(gpu_patch)

    proposal = mini_scene.get_proposal()
    proposer = Proposer(p)
    chi2, chi2_derivs, residuals = proposer.evaluate_proposal(proposal)

    print('Plotting residuals...')
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    n_exp_max = np.max(p.n_exp)

    figsize = np.array([p.n_bands, n_exp_max])
    fig, axes = plt.subplots(p.n_bands, n_exp_max, sharex=True, sharey=True, dpi=72, figsize=figsize*4)
    for b in range(p.n_bands):
        axrow = axes[b]
        for e in range(p.band_N[b]):
            ax = axrow[e]
            ax.set_aspect('equal')
            ax.imshow(residuals[0])
    #fig.colorbar()
    fig.savefig(pjoin(scratch_dir, 'residual0.png'))
