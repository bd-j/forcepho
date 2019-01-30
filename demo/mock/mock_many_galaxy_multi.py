# --------
# Basic script for fitting mock images with simple single-gaussian sources and
# PSFs
# ---------

import sys, os, time
import argparse
import numpy as np
import matplotlib.pyplot as pl

from forcepho import paths
from forcepho.sources import SimpleGalaxy, Galaxy, Scene
from forcepho.likelihood import WorkPlan, make_image
from forcepho import fitting

from demo_utils import make_stamp, numerical_image_gradients
from phoplot import display, plot_model_images


# --------------------------------
# --- Command Line Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=float, nargs='*', default=[30, 30],
                    help="size in pixels of cutout")
parser.add_argument("--scene_file", type=str, default="catalog_for_demo.dat",
                    help="file containing scene specification")
parser.add_argument("--sersic", action="store_true",
                    help="If set, mock and fit as Sersics, otherwise as gaussians with fixed size")
parser.add_argument("--peak_snr", type=float, default=10,
                    help="S/N of brightest pixel")
parser.add_argument("--add_noise", action="store_false",
                    help="Whether to add noise to the mock")
parser.add_argument("--backend", type=str, default="none",
                    help="Sampling backend to use")
parser.add_argument("--jitter", type=float, default=0.1,
                    help="perturb initial guess by this fraction")
parser.add_argument("--nwarm", type=int, default=1000,
                    help="number of iterations for hemcee burn-in")
parser.add_argument("--niter", type=int, default=500,
                    help="number of iterations for hemcee production")
parser.add_argument("--nlive", type=int, default=-1,
                    help="number of dynesty live points")
parser.add_argument("--results_name", type=str, default="demo_mock_many_gal_multi",
                    help="root name and path for the output pickle.'none' results in no output.")
parser.add_argument("--display", action="store_true",
                    help="Whether to plot fit information after fitting")
parser.add_argument("--plot_dir", type=str, default="",
                    help="Where to save plots of fit infomation.  If empty, plots are not saved")
parser.add_argument("--show_model", action="store_true",
                    help="Whether to plot the model")
parser.add_argument("--test_grad", action="store_true",
                    help="Whether to compare gradient to numerical gradients")


psfnames = {"F090W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F115W": os.path.join(paths.psfmixture, 'f090_ng6_em_random.p'),
            "F150W": os.path.join(paths.psfmixture, 'f150w_ng6_em_random.p'),
            "F200W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            "F277W": os.path.join(paths.psfmixture, 'f200w_ng6_em_random.p'),
            }


def setup_scene(sourceparams=[(1.0, 5., 5., 0.7, 30., 1.0, 0.05)],
                add_noise=False, peak_snr=100.,
                splinedata=None, filters=['dummy'], stamp_kwargs=[]):

    # --- Get Sources and a Scene -----
    sources = []
    for pars in sourceparams:
        flux, x, y, q, pa, n, rh = np.copy(pars)
        if splinedata is not None:
            s = Galaxy(filters=filters, splinedata=splinedata)
            s.sersic = n
            s.rh = rh
        else:
            s = SimpleGalaxy(filters=filters)
        s.flux = flux
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    scene = Scene(sources)
    theta = scene.get_all_source_params()

    # --- Get Stamps ----
    print(len(stamp_kwargs))
    stamps = [make_stamp(**sk) for sk in stamp_kwargs]
    if splinedata is not None:
        for stamp in stamps:
            stamp.scale = np.array([[32.0, 0.0], [0.0, 32.0]])

    # --- Generate mock and add to stamp ---
    ptrue = np.array(theta).copy()
    for stamp in stamps:
        true_image, _ = make_image(scene, stamp, Theta=ptrue)
        stamp.pixel_values = true_image.copy()
        err = stamp.pixel_values.max() / peak_snr
        #err = np.sqrt(err**2 + stamp.pixel_values.flatten())
        err = np.ones(stamp.npix) * err
        stamp.ierr = np.ones(stamp.npix) / err
        if add_noise:
            noise = np.random.normal(0, err)
            stamp.pixel_values += noise.reshape(stamp.nx, stamp.ny)
            
    return scene, stamps, ptrue


def get_bounds(scene, npix=3, maxfluxfactor=20., plate_scale=[1., 1.]):
    #plate_scale = np.abs(np.linalg.eigvals(np.linalg.inv(stamps[0].dpix_dsky)))
    npix = 3.0
    try:
        sersic_low = [scene.sources[0].sersic_range[0], scene.sources[0].rh_range[0]]
        sersic_hi = [scene.sources[0].sersic_range[-1], scene.sources[0].rh_range[-1]]
    except(AttributeError):
        sersic_low = []
        sersic_hi = []
        
    lower = [s.nband * [0.] +
             [s.ra - npix * plate_scale[0], s.dec - npix * plate_scale[1],
              0.3, -np.pi/1.5] + sersic_low
             for s in scene.sources]
    upper = [(np.array(s.flux) * maxfluxfactor).tolist() +
             [s.ra + npix * plate_scale[0], s.dec + npix * plate_scale[1],
              1.0, np.pi/1.5] + sersic_hi
             for s in scene.sources]

    lower = np.concatenate(lower)
    upper = np.concatenate(upper)
    return (lower, upper)


def read_scene_catalog(catname):
    with open(catname, "r") as f:
        lines = f.readlines()
    nband = int(lines[0].replace('#', '').split()[1])
    filters = lines[1].replace('#', '').split()[:nband]
    filters = [f.upper() for f in filters]
    i = 2
    dat = []
    while lines[i][0] != '#':
        dat.append(np.array([float(c) for c in lines[i].split()]))
        i += 1
    #dat = np.array(dat)
    i += 1
    offsets = []
    while (lines[i][0] != '#'):
        offsets.append([float(c) for c in lines[i].split()])
        i += 1
    offsets = np.array(offsets)

    # flux, ra, dec, q, pa(deg), n, sersic
    params = [(d[:nband], d[nband], d[nband+1], d[nband+2], d[nband+3], 0.0, 0.0)
              for d in dat if len(d) >= (nband + 4)]

    return params, filters, offsets   


if __name__ == "__main__":

    args = parser.parse_args()
    plotname = os.path.basename(args.results_name)  # filename for plots
    if args.sersic:
        splinedata = paths.galmixtures[1]
    else:
        splinedata = None

    # --- Setup Scene and Stamp(s) ---
    sourcepars, filters, offsets = read_scene_catalog(args.scene_file)    
    stamp_kwargs = [{'size': tuple(args.size), 'psfname': psfnames[f], 'filtername': f, 'offset': o}
                    for o in offsets for f in filters]

    scene, stamps, ptrue = setup_scene(sourceparams=sourcepars, splinedata=splinedata,
                                       peak_snr=args.peak_snr, add_noise=args.add_noise,
                                       filters=filters, stamp_kwargs=stamp_kwargs)
    lower, upper = get_bounds(scene, npix=2)
    plans = [WorkPlan(stamp) for stamp in stamps]

    # --- show model ----
    if args.show_model:
        rfig, raxes = pl.subplots(len(filters), len(offsets), sharex=True, sharey=True)
        raxes = np.atleast_2d(raxes)
        for i, stamp in enumerate(stamps):
            raxes.flat[i].imshow(stamp.pixel_values.T, origin='lower')
        pl.show()

    # --- Gradient Check ---
    if args.test_grad:
        delta = np.ones_like(ptrue) * 1e-7
        #numerical
        grad_num = numerical_image_gradients(ptrue, delta, scene, stamp)
        image, grad = make_image(scene, stamps[0], Theta=ptrue)
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
        p0 = ptrue * np.random.normal(1.0, args.jitter, size=ptrue.shape)
        result = fitting.run_opt(p0, scene, plans, jac=True)
        result.args = args
        result.stamps = [stamp]
        plot_model_images(result.chain[-1], result.scene, result.stamps)


    # --- Sampling ---
    if args.backend == "pymc3":
        result = fitting.run_pymc3(ptrue.copy(), scene, plans, lower=lower, upper=upper,
                                    nwarm=args.nwarm, niter=args.niter)

        result.stamps = stamps
        result.args = args
        _ = display(result, savedir=args.plot_dir, show=args.display, root=plotname)

    # --- No fitting ---
    if args.backend == "none":
        sys.exit()        

    pl.show()
