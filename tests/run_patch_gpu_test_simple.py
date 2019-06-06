#!/usr/bin/env python

import os
import sys
sys.path.append('..')

import numpy as np

from forcepho.patch import Patch
from forcepho.proposal import Proposer

from patch_conversion import patch_conversion

import pycuda.autoinit

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
    proposer.evaluate_proposal(proposal)
