# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys, os
import numpy as np
import matplotlib.pyplot as pl

from forcepho.sources import Star, SimpleGalaxy, Scene
from forcepho.likelihood import WorkPlan, make_image
from forcepho import fitting

from demo_utils import make_stamp, numerical_image_gradients


# --------------------------------
# --- Command Line Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=float, nargs='*', default=[30, 30],
                    help="size in pixels of cutout")
parser.add_argument("--fwhm", type=float, default=2.0,
                    help="FWHM of the PSF in pixels")
parser.add_argument("--source_type", type=str, default="galaxy",
                    help="Source type, galaxy | star")
parser.add_argument("--add_noise", action="store_false",
                    help="Whether to add noise to the mock")
parser.add_argument("--nwarm", type=int, default=1000,
                    help="number of iterations for hemcee burn-in")
parser.add_argument("--niter", type=int, default=500,
                    help="number of iterations for hemcee production")
parser.add_argument("--nlive", type=int, default=-1,
                    help="number of dynesty live points")
parser.add_argument("--backend", type=str, default="none",
                    help="Sampling backend to use")
parser.add_argument("--results_name", type=str, default="demo_mock_one_gauss_single_gpsf",
                    help="root name and path for the output pickle.'none' results in no output.")
parser.add_argument("--display", action="store_true",
                    help="Whether to plot fit information after fitting")
parser.add_argument("--plot_dir", type=str, default="",
                    help="Where to save plots of fit infomation.  If empty, plots are not saved")
parser.add_argument("--show_grad", action="store_true",
                    help="Whether to plot gradients of the model")
parser.add_argument("--test_grad", action="store_true",
                    help="Whether to compare gradient to numerical gradients")



def setup_scene(galaxy=False, fwhm=1.0, offset=0.0,
                size=(30, 30), add_noise=False):

    stamp = make_stamp(size, fwhm, offset=offset)

    # --- Get a Source and Scene -----
    if galaxy:
        ngauss = 1
        source = SimpleGalaxy()
        source.radii = np.arange(ngauss) * 0.5 + 1.0
        source.q = 0.5
        source.pa = np.deg2rad(30.)
        theta = [100., 10., 10., 0.5, np.deg2rad(10.)]
        label = ['$\psi$', '$x$', '$y$', '$q$', '$\\varphi$']
        bounds = [(0, 1e4), (0., 30), (0, 30), (0, 1), (0, np.pi/2)]
    else:
        source = Star()
        theta = [100., 10., 10.]
        label = ['$\psi$', '$x$', '$y$']
        bounds = [(-1e6, 1e6), (-1e5, 1e5), (-1e5, 1e5)]

    scene = Scene([source])

    # --- Generate mock  and add to stamp ---
    ptrue = np.array(theta)
    true_image, partials = make_image(scene, stamp, Theta=ptrue)
    stamp.pixel_values = true_image.copy()
    err = stamp.pixel_values.max() * 1e-2
    #err = np.sqrt(err**2 + stamp.pixel_values.flatten())
    err = np.ones(stamp.npix) * err
    stamp.ierr = np.ones(stamp.npix) / err

    if add_noise:
        noise = np.random.normal(0, err)
        stamp.pixel_values += noise.reshape(stamp.nx, stamp.ny)

    return scene, stamp, ptrue, label


if __name__ == "__main__":

    args = parser.parse_args()
    galaxy = args.source_type == "galaxy"

    # Get a scene and a stamp at some parameters
    blob = setup_scene(galaxy=galaxy, fwhm=args.fwhm, add_noise=args.add_noise)
    scene, stamp, ptrue, label = blob
    true_image, partials = make_image(scene, stamp, Theta=ptrue)
    plans = [WorkPlan(stamp)]

    # ---- Plot mock image and gradients thereof -----
    if args.show_grad:
        fig, axes = pl.subplots(3, 2)
        for i, ddtheta in enumerate(partials):
            ax = axes.flat[i+1]
            c = ax.imshow(ddtheta.reshape(stamp.nx, stamp.ny).T, origin='lower')
            ax.text(0.1, 0.85, '$\partial I/\partial${}'.format(label[i]), transform=ax.transAxes)
            fig.colorbar(c, ax=ax)

        ax = axes.flat[0]
        c = ax.imshow(true_image.T, origin='lower')
        ax.text(0.1, 0.85, 'Mock (I)'.format(label[i]), transform=ax.transAxes)
        fig.colorbar(c, ax=ax)


    # ---- Test Image Gradients ------
    if args.test_grad:
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


    # --- Optimization -----
    if args.backend == "opt":
        result = fitting.run_opt(ptrue.copy() * args.jitter,
                                 scene, plans, jac=True)
        result.args = args
        result.stamps = stamps

    # sampling
    if args.backend == "pymc3":
        result = backends.run_pymc3(p0, scene, plans, lower=lower, upper=upper,
                                    nwarm=args.nwarm, niter=args.niter)

        result.stamps = stamps
        result.args = args


    # --- No fitting ---
    if args.backend == "none":
        sys.exit()

    # --- Plotting
    _ = display(result, savedir=args.plot_dir, show=args.display, root=args.results_name)
    pl.show()
    
