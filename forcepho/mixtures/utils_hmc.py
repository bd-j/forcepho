# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np

try:
    from jax import random
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoDelta
    import numpyro.optim as optim
except(ImportError):
    pass

Image = namedtuple("Image", "xpix ypix data unc nx ny cx cy")


def infer(model, hmc=True, num_warmup=1000, num_samples=500, dense_mass=True,
          MAP=False, max_tree_depth=10, **model_args):
    """Infer the parameters of a gaussian mixture using HMC or SVI (experimental).

    Parameters
    ----------
    model : callable
        A numpyro model specification that accepts arguments supplied here as
        extra keyword arguments

    Returns
    -------
    best : dictionary
        A dictionary of best fit model parameters (some values of which may be
        vectors if multiple gaussians are used), keyed by parameter name.

    samples : dictionary
        A dictionary of all posterior samples as ndarrays, keyed by parameter
        name.  Vector parameters have shape (nsamples, npar)

    mcmc : numpyro.infer.MCMC() instance
        The MCMC object, including posterior samples, which can be input into say,
        arviz functions.
    """
    if hmc:
        kernel = NUTS(model, dense_mass=dense_mass, max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)
        mcmc.run(rng_key_, extra_fields=("potential_energy", "accept_prob"), **model_args)

        samples = mcmc.get_samples()
        nlnp = mcmc.get_extra_fields()["potential_energy"]
        ind_best = nlnp.argmin()
        best = {k: samples[k][ind_best] for k in samples.keys()}

        return best, samples, mcmc

    else:
        # NOTE: !!! DOES NOT WORK (out of the box)
        raise NotImplementedError
        if MAP:
            guide = AutoDelta(model)
        else:
            guide = AutoLaplaceApproximation(model)

        svi = SVI(model,
                  guide,
                  optim.Adam(0.001),
                  loss=Trace_ELBO(),
                  **model_args)

        svi_result = svi.run(random.PRNGKey(0), 2000)
        params = svi_result.params

        #init_state = svi.init(random.PRNGKey(0))
        #state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
        #params = svi.get_params(state)
        samples = guide.sample_posterior(random.PRNGKey(1), params, (num_samples,))
        best = {k: samples[k].mean(axis=0) for k in samples.keys()}
        post = guide.get_posterior(params)

        return best, samples, (guide, params)


def show_exp(xpix, ypix, value, ax=None, **imshow_kwargs):
    """Create a rectangular image that bounds the given pixel coordinates
    and assign `value` to the correct pixels. Pixels in the rectangle that do
    not have assigned values are given nan.  use imshow to display the image in
    standard astro format (x increasing left to right, y increasing bottom to
    top)
    """
    lo = np.array((xpix.min(), ypix.min())) - 0.5
    hi = np.array((xpix.max(), ypix.max())) + 0.5
    size = hi - lo
    im = np.zeros(size.astype(int)) + np.nan

    x = (xpix-lo[0]).astype(int)
    y = (ypix-lo[1]).astype(int)
    # This is the correct ordering of xpix, ypix subscripts
    im[x, y] = value

    if ax is not None:
        # note transpose and origin to 'properly' display x-axis as leading dimension
        cb = ax.imshow(im.T, origin="lower",
                       extent=(lo[0], hi[0], lo[1], hi[1]),
                       **imshow_kwargs)
    return im, (ax, cb)


def display(model, image, figsize=(10, 4)):
    """
    Parameters
    ----------
    model : ndarray of shape (npix,)

    image : object
        Must have the attributes `xpix`, `ypix`, `data`, each of which are ndarrays of shape (npix,)
    """
    import matplotlib.pyplot as pl
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import get_cmap
    cmap = get_cmap('viridis')

    fig = pl.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[1, 15])
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cbars = [fig.add_subplot(gs[0, i]) for i in range(3)]

    i = 0
    im, (ax, cb) = show_exp(image.xpix, image.ypix, image.data, ax=axes[i], cmap=cmap)
    _ = pl.colorbar(cb, cax=cbars[i], orientation="horizontal",
                    label=r"Input")

    i = 1
    ii, (ax, cb) = show_exp(image.xpix, image.ypix, model, ax=axes[i], cmap=cmap)
    _ = pl.colorbar(cb, cax=cbars[i], orientation="horizontal",
                    label=r"Best Model")

    i = 2
    #cmap = get_cmap("Spectral")
    chi = (model - image.data) / (image.data.max() * 0.01)
    mm = np.max(np.abs(chi))
    im, (ax, cb) = show_exp(image.xpix, image.ypix, chi, ax=axes[i], vmin=-mm, vmax=mm, cmap=cmap)
    _ = pl.colorbar(cb, cax=cbars[i], orientation="horizontal",
                    label=r"Model-Data as % of peak")

    print(chi.min(), chi.max())
    return fig, axes


def radial_plot(image, model, times_r=True):
    r = np.hypot(image.xpix-image.cx, image.ypix-image.cy)

    import matplotlib.pyplot as pl
    fig, axes = pl.subplots(2, sharex=True)
    ax = axes[0]
    ax.plot(r, image.data, ".", color="royalblue", label=r"data", rasterized=True)
    ax.plot(r, model, ".", color="darkorange", label=r"model", rasterized=True)
    ax.legend()
    #ax.set_xscale("log")
    ax.set_ylim(max(model.min(), image.data.min()) * 0.1, image.data.max() * 2)
    ax.set_yscale("log")

    if times_r:
        factor = r
    else:
        factor = 1

    ax = axes[1]
    ax.plot(r, factor * (model-image.data), ".", color="firebrick", rasterized=True, label=r"(model - data) $\times \, r$")
    ax.axhline(image.data.max() * 0.01, linestyle=":", color="k", label=r"1% of max pixel")
    ax.axhline(0, linestyle="--", color="k")
    ax.legend()

    ax.set_xlabel(r"$r$ (pixels)")  # for ax in axes]

    return fig, axes


def draw_ellipses(best, ax, cmap=None):
    from matplotlib.patches import Ellipse
    if cmap is None:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('viridis')

    ngauss = len(best["x"])
    for i in range(ngauss):
        # need to swap axes here, not sure why
        mu = np.array([best["x"][i], best["y"][i]])
        sy = best["sx"][i]
        sx = best["sy"][i]
        sxy = best["rho"][i] * sx * sy
        # construct covar matrix and get eigenvalues
        S = np.array([[sx**2, sxy],[sxy, sy**2]])
        vals, vecs = np.linalg.eig(S)
        # get ellipse params
        theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
        w, h = 2 * np.sqrt(vals)
        ell = Ellipse(xy=mu, width=w, height=h, angle=theta)
        ax.add_artist(ell)
        #e.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ell.set_facecolor(cmap(best["weight"][i]))

    ax.set_xlim(0, 2 * best["x"].max())
    ax.set_ylim(0, 2 * best["y"].max())

    return ax


if __name__ == "__main__":
    pass
