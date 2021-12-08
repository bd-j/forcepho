#!/usr/bin/env python

from time import time
import os, sys
sys.path.append('..')

import numpy as np

from forcepho.sources import Scene
from forcepho.likelihood import WorkPlan, lnlike_multi, make_image

from patch_conversion import patch_conversion, zerocoords


if __name__ == '__main__':
    #path_to_data = '/gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data'
    path_to_data = "test_data/"
    patch_name = os.path.join(path_to_data, "test_patch_mini.h5")
    splinedata = os.path.join(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = path_to_data
    nradii = 9

    list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)
    zerocoords(list_of_stamps, mini_scene)
    #sys.exit()
    
    #sublist_of_stamps = [s for s in list_of_stamps if s.filtername == "f814w"] 
    plans = [WorkPlan(stamp) for stamp in list_of_stamps]
    theta = mini_scene.get_all_source_params().copy()
    t = time()
    lnp, lnp_grad = lnlike_multi(theta, mini_scene, plans)
    dur = time() - t

    scene =  mini_scene
    stamps = list_of_stamps

    npix = np.array([s.npix for s in stamps])
    nband = len(scene.sources[0].filternames)
    npsf = np.array([len(scene.sources[0].psfgauss(e)) 
                     for e in range(len(stamps))])
    nsource = len(scene)

    print(dur)
    s0 = mini_scene.sources[0]
    scene = Scene([s0])
    stamp0 = list_of_stamps[0]
    images, grads = make_image(scene, stamp0)
    
    sys.exit()
    
    import matplotlib.pyplot as pl
    fig, axes = pl.subplots(len(grads), 1, sharex=True, sharey=True)
    for g, ax in zip(grads, axes):
        c = ax.imshow(g.reshape(stamp0.shape).T, origin="lower")
        pl.colorbar(c, ax=ax)
        
        print(g.reshape(stamp0.shape)[110, 5])

    pl.show()
