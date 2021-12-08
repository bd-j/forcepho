#------------
# script with utilities to numerically test gradients
# -----------
import os
import numpy as np
import matplotlib.pyplot as pl
from forcepho import gaussmodel as gm
from forcepho.sources import Galaxy, Scene
from forcepho.stamp import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.likelihood import lnlike_multi, make_image, WorkPlan
from patch_conversion import patch_conversion

from forcepho.gaussmodel import convert_to_gaussians
#from forcepho.gaussmodel import fast_dot_dot_2x2, fast_matmul_matmul_2x2, fast_trace_2x2

splinedata = "test_data/sersic_mog_model.smooth=0.0150.h5"


def numerical_image_gradients(theta0, delta, scene=None, stamp=None):

    dI_dp = []
    for i, dp in enumerate(delta):
        theta = theta0.copy()
        theta[i] -= dp
        imlo, _ = make_image(scene, stamp, Theta=theta)
        theta[i] += 2 *dp
        imhi, _ = make_image(scene, stamp, Theta=theta)
        dI_dp.append((imhi - imlo) / (2 *dp))

    return np.array(dI_dp)


def test_image_gradients(ptrue, delta=None, scene=None, stamp=None):
    delta = np.ones_like(ptrue) * 1e-6
    #numerical
    grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
    image, grad = make_image(scene, stamp, Theta=ptrue)
    fig, axes = pl.subplots(len(ptrue), 3, sharex=True, sharey=True)
    for i in range(len(ptrue)):
        g = grad[i,:].reshape(stamp.nx, stamp.ny)
        c = axes[i, 0].imshow(grad_num[i,:,:].T, origin='lower')
        fig.colorbar(c, ax=axes[i, 0])
        c = axes[i, 1].imshow(g.T, origin='lower')
        fig.colorbar(c, ax=axes[i, 1])
        c = axes[i, 2].imshow((grad_num[i,:,:] - g).T, origin='lower')
        fig.colorbar(c, ax=axes[i, 2])

    axes[0, 0].set_title('Numerical')
    axes[0, 1].set_title('Analytic')
    axes[0, 2].set_title('N - A')




if __name__ == "__main__":

    #path_to_data = '/gpfs/wolf/gen126/proj-shared/jades/patch_conversion_data'
    path_to_data = "test_data/"
    patch_name = os.path.join(path_to_data, "test_patch_mini.h5")
    splinedata = os.path.join(path_to_data, "sersic_mog_model.smooth=0.0150.h5")
    psfpath = path_to_data
    nradii = 9

    stamps, scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)
    
    ptrue = scene.get_all_source_params().copy()

    test_image_gradients(ptrue, delta=None, scene=scene, stamp=stamps[0])
    pl.show()
    
    plans = [WorkPlan(stamp) for stamp in stamps]
    lnp, lnp_grad = lnlike_multi(ptrue, scene, plans)
    
    plan = plans[0]
    source = scene.sources[0]
    gig = convert_to_gaussians(source, plan.stamp)
    
    #fig, ax, grad, grad_num = test_image_gaussian_gradients(1e-7)
