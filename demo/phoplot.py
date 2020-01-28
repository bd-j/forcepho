import os, sys, glob
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import cPickle as pickle

from forcepho.likelihood import make_image

# code to force angular values to an interval
# phim = np.mod((phi + np.pi / 2), np.pi) - np.pi / 2.


def display(data, savedir="", show=False, root="xdf", **imkwargs):

    if type(data) is str:
        fn = data
        root = fn.split("/")[-1].replace('.pkl', '')
        with open(fn, "r") as f:
            try:
                result = pickle.load(f)
            except(ImportError):
                print("couldn't import {}".format(fn))
                return
    else:
        result = data

    if (show is False) & (savedir == ""):
        return result

    # --- Chains ---
    npar = np.max([s.nparam for s in result.scene.sources])
    fig, axes = pl.subplots(npar + 1, len(result.scene.sources),
                            sharex=True, figsize=(13, 12))
    for i, ax in enumerate(axes[:-1, ...].T.flat):
        ax.plot(result.chain[:, i])
        ax.set_ylabel(result.scene.parameter_names[i])
    try:
        [ax.plot(result.lnp) for ax in axes[-1, ...].flat]
        [ax.set_xlabel("iteration") for ax in axes[-1, ...].flat]
        axes[-1, ...].set_ylabel("ln P")
    except(AttributeError):
        [ax.set_xlabel("iteration") for ax in axes[-2, ...].flat]
        [ax.set_visible(False) for ax in axes[-1, ...].flat]
    #for i, ax in enumerate(axes.T.flat): ax.axhline(result.pinitial[i], color='k', linestyle=':')
    if savedir != "":
        fig.savefig(os.path.join(savedir, root + ".chain.pdf"))

    # --- Corner Plot ----
    import corner
    cfig = corner.corner(result.chain, labels=result.scene.parameter_names,
                         show_titles=True, fill_contours=True,
                         plot_datapoints=False, plot_density=False)
    if savedir != "":
        cfig.savefig(os.path.join(savedir, root + ".corner.pdf"))

    # --- Residual ---
    try:
        best = result.chain[result.lnp.argmax(), :]
    except:
        best = result.chain[-1, :]
        print("using last position")
    rfig, raxes = plot_model_images(best, result.scene, result.stamps,
                                    share=False, **imkwargs)
    if savedir != "":
        rfig.savefig(os.path.join(savedir, root + ".residual.pdf"))

    if show:
        pl.show()
    else:
        pl.close(rfig)
        pl.close(fig)
        pl.close(cfig)

    return result


def plot_model_images(pos, scene, stamps, axes=None, colorbars=True,
                      x=slice(None), y=slice(None), share=True,
                      scale_model=False, scale_residuals=False):
    vals = pos
    same_scale = [False, scale_model, scale_residuals, False]
    if axes is None:
        figsize = (14, 3.3 * len(stamps) + 0.5)
        rfig, raxes = pl.subplots(len(stamps), 4, figsize=figsize,
                                  sharex=share, sharey=share)
    else:
        rfig, raxes = None, axes
    raxes = np.atleast_2d(raxes)
    for i, stamp in enumerate(stamps):
        data = stamp.pixel_values
        im, grad = make_image(scene, stamp, Theta=vals)
        resid = data - im
        chi = resid * stamp.ierr.reshape(stamp.nx, stamp.ny)
        ims = [data, im, resid, chi]
        norm = None
        for j, ii in enumerate(ims):
            if same_scale[j] and colorbars:
                vmin, vmax = cb.vmin, cb.vmax
            else:
                vmin = vmax = None
            ci = raxes[i, j].imshow(ii[x, y].T, origin='lower', vmin=vmin, vmax=vmax)
            if (rfig is not None) & colorbars:
                cb = rfig.colorbar(ci, ax=raxes[i, j], orientation='horizontal')
                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=-55)
        text = "{}\n({}, {})".format(stamp.filtername, stamp.crval[0], stamp.crval[1])
        ax = raxes[i, 1]
        ax.text(0.6, 0.1, text, transform=ax.transAxes, fontsize=10)

    labels = ['Data', 'Model', 'Data-Model', "$\chi$"]
    [ax.set_title(labels[i]) for i, ax in enumerate(raxes[0, :])]
    return rfig, raxes


def load_results(fn="sampler_demo_semi_v1.pkl"):
    with open(fn, "rb") as f:
        sampler = pickle.load(f)

    scene = sampler.scene
    plans = sampler.plans
    return sampler, scene, plans


def plot_data(stamps, filters, offsets):
    figsize = (len(offsets) * 1.5 + 1.0, len(filters) * 1.5 + 1.0)
    dfig, daxes = pl.subplots(len(filters), len(offsets), figsize=figsize,
                              sharex=True, sharey=True)
    daxes = np.atleast_2d(daxes)
    for i, stamp in enumerate(stamps):
        daxes.flat[i].imshow(stamp.pixel_values.T, origin='lower')

    [ax.set_title("(dx, dy)=\n({}, {})".format(*o), fontsize=12)
     for (o, ax) in zip(offsets, daxes[0, :])]
    [ax.set_ylabel("{}".format(f)) for (f, ax) in zip(filters, daxes[:, 0])]
    return dfig, daxes


def plot_chain(sampler, start=0, stop=-1, source=0,
               show_trajectories=False, equal_axes=False):
    ndim = sampler.scene.sources[0].nparam
    p0 = sampler.truths
    filters = sampler.filters
    nband = len(filters)

    ffig, faxes = pl.subplots(nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios': [2, 1]},
                              sharex='col', sharey='row')
    faxes = np.atleast_2d(faxes)
    for j in range(nband):
        i = source * ndim + j
        _ = one_chain(faxes[j, :], sampler, i, p0[i], start, stop,
                      show_trajectories=show_trajectories)

    faxes = prettify_chains(faxes, filters, equal_axes=equal_axes,
                            show_trajectories=show_trajectories)

    tfig, taxes = pl.subplots(ndim - nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios': [2, 1]})
    for j in range(ndim - nband):
        i = source * ndim + nband + j
        _ = one_chain(taxes[j, :], sampler, i, p0[i], start, stop,
                      show_trajectories=show_trajectories)

    pnames = ['RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)', 'n', 'r$_h$']
    taxes = prettify_chains(taxes, pnames, equal_axes=equal_axes,
                            show_trajectories=show_trajectories)

    return tfig, taxes, ffig, faxes


def one_chain(axes, sampler, i, truth=None, start=0, stop=-1,
              show_trajectories=False, full_hist=False):
    l, n = [], 0
    if show_trajectories:
        label = "trajectories"
        for k, traj in enumerate(sampler.trajectories[:stop]):
            if sampler.accepted[k]:
                color = 'tomato'
            else:
                color = 'maroon'
            axes[0].plot(np.arange(len(traj)) + n,
                         traj[:, i], color=color, label=label)
            n += len(traj)
            l.append(n)
            label = None
        l = np.array(l) - 1
        cut = l[start]
    else:
        l = np.arange(sampler.chain.shape[0])[:stop]
        cut = start
    axes[0].plot(l, sampler.chain[:stop, i], 'o', label="samples")
    if full_hist:
        axes[1].hist(sampler.chain[start:, i], alpha=0.5, bins=30,
                     color='grey', orientation='horizontal')
    axes[1].hist(sampler.chain[start:stop, i], alpha=0.5, bins=30,
                 orientation='horizontal')

    if start > 0:
        axes[0].axvline(cut, linestyle='--', color='r')
    if truth is not None:
        axes[0].axhline(truth, linestyle=':', color='k', label="Truth")
        axes[1].axhline(truth, linestyle=':', color='k')
    return axes


def prettify_chains(axes, labels, fontsize=10, equal_axes=False,
                    show_trajectories=False):
    [ax.set_ylabel(p) for ax, p in zip(axes[:, 0], labels)]
    [ax.set_xticklabels('') for ax in axes[:-1, 0]]
    [ax.set_xticklabels('') for ax in axes[:, 1]]
    [ax.yaxis.tick_right() for ax in axes[:, 1]]
    [ax.set_ylabel(p) for ax, p in zip(axes[:, 1], labels)]
    [ax.yaxis.set_label_position("right") for ax in axes[:, 1]]
    if show_trajectories:
        axes[-1, 0].set_xlabel('Step number')
        #axes[0, 0].legend()
    else:
        axes[-1, 0].set_xlabel('HMC iteration')

    [item.set_fontsize(fontsize)
     for ax in axes.flat
     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels())
    ]

    if equal_axes:
        [ax[0].set_ylim(ax[1].get_ylim()) for ax in axes]

    return axes


def plot_seds(sampler, start=0):
    truth = sampler.sourcepars
    filternames = sampler.filters
    nband = len(filternames)
    ndim = sampler.scene.sources[0].nparam

    from sedpy.observate import load_filters
    filt = load_filters(['jwst_{}'.format(f.lower()) for f in filternames])
    wave = np.array([f.wave_effective for f in filt]) / 1e4

    sfig, saxes = pl.subplots(len(truth), 1, sharex=True)
    for isource, source in enumerate(truth):
        ax = saxes.flat[isource]
        fluxes = source[0]
        s = ndim * isource
        samples = sampler.chain[start:, s:(s + nband)]
        pct = np.percentile(samples, [16, 50, 84], axis=0)
        neg = pct[1, :] - pct[0, :]
        plus = pct[2, :] - pct[1, :]
        print("{}: {}".format(isource, (neg + plus) / pct[1, :]))
        ax.errorbar(wave, pct[1, :], yerr=[neg, plus], label="Posterior",
                    capsize=5.0, capthick=2.0)
        ax.plot(wave, fluxes, '-o', label="Truth".format(isource))
        ax.text(0.5, 0.75, "Source #{}".format(int(isource + 1)),
                transform=ax.transAxes, fontsize=10)

    [ax.set_ylim(*(np.array(ax.get_ylim()) * [0.8, 1.2])) for ax in saxes.flat]
    [ax.set_xlim(0.8, 2.9) for ax in saxes.flat]
    [ax.set_ylabel("Flux") for ax in saxes.flat]
    saxes.flat[-1].set_xlabel("$\lambda (\mu m)$")
    return sfig, saxes


def hmc_movie(sampler, iterations, stamps, pars_to_show, outname="movie_test.pdf",
              label=[], traj=True, x=slice(None), y=slice(None), png=False):

    maxstep = np.sum([len(sampler.trajectories[i])
                      for i in range(np.max(iterations))])
    pinds = pars_to_show
    ranges = ranges = np.array([sampler.chain.min(axis=0), sampler.chain.max(axis=0)]).T

    if png:
        dirn = os.path.dirname(outname)
        fn = os.path.basename(outname).split('.')[0]
        dirn = os.path.join(dirn, "{}_movie/".format(fn))
        fn = "s"
        try:
            os.mkdir(dirn)
        except:
            pass
    else:
        pdf = PdfPages(outname)
    for k, i in enumerate(iterations):
        fig, axlist = hmc_movie_frame(sampler, i, stamps, x=x, y=y,
                                      show_trajectories=traj,
                                      show=pinds, ranges=ranges[pinds, :])
        saxes, caxes, haxes = axlist
        paxes = prettify_chains(np.vstack([caxes, haxes]).T, label,
                                show_trajectories=traj)
        [ax.set_xlim(0, maxstep) for ax in caxes]
        sz = 3.0
        fig.set_size_inches(2.0 + 3./2. * (sz * 3), sz * len(stamps) + 0.5)
        if png:
            fig.savefig("{}/{}-{:02.0f}.png".format(dirn, fn, k), dpi=500)
        else:
            pdf.savefig(fig)
        pl.close(fig)

    pdf.close()


def hmc_frame_geometry(nstamp, npar):

    fig = pl.figure()
    gs0 = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nstamp, 3, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(npar, 3, subplot_spec=gs0[1])

    saxes = [pl.Subplot(fig, gs00[i, j]) for i in range(nstamp) for j in range(3)]
    [fig.add_subplot(ax) for ax in saxes]
    saxes = np.array(saxes).reshape(nstamp, 3)

    caxes = np.array([pl.Subplot(fig, gs01[j, :2]) for j in range(npar)])
    haxes = np.array([pl.Subplot(fig, gs01[j, 2]) for j in range(npar)])
    [fig.add_subplot(ax) for ax in caxes]
    [fig.add_subplot(ax) for ax in haxes]
    return fig, [saxes, caxes, haxes]


def hmc_movie_frame(sampler, i, stamps, show=[0], ranges=None,
                    show_trajectories=True, x=slice(None), y=slice(None)):

    nstamp = len(stamps)
    npar = len(show)
    fig, [saxes, caxes, haxes] = hmc_frame_geometry(nstamp, npar)
    pos = sampler.chain[i, :]
    # --- Show data, model and residual
    _, saxes = plot_model_images(pos, sampler.scene, stamps, axes=saxes,
                                 x=x, y=y)
    # --- Show chain up to point i
    for j, paridx in enumerate(show):
        _, axes = one_chain([caxes[j], haxes[j]], sampler, paridx, stop=i,
                            show_trajectories=show_trajectories, full_hist=True)
        if ranges is not None:
            [ax.set_ylim(*ranges[j]) for ax in [caxes[j], haxes[j]]]

    return fig, [saxes, caxes, haxes]


if __name__ == "__main__":

    fn = "mock_simplegm_20stamp_snr05.pkl"
    to_show = [0, 4, 8, 12, 16]
    iterations = np.arange(1, 100)  # [1, 2, 3, 4, 5] #, 10, 15, 20]
    psub = np.array([0, 2, 5, 6, 7, 8])
    stop = 100
    start = 0
    sourceid = 1

    sampler, scene, plans = load_results(fn=fn)
    stamps = [p.stamp for p in plans]
    stamps_to_show = [stamps[i] for i in to_show]
    pos = sampler.chain[sampler.lnp.argmax(), :]

    if False:
        dfig, daxes = plot_data(stamps, sampler.filters, sampler.offsets)
        dfig.savefig(os.path.join('figures', fn.replace(".pkl", "_data.pdf")))

        sfig, saxes = plot_seds(sampler, start=start)
        sfig.savefig(os.path.join('figures', fn.replace(".pkl", "_seds.pdf")))

        rfig, raxes = plot_model_images(pos, scene, stamps_to_show)
        rfig.savefig(os.path.join('figures', fn.replace(".pkl", "_residuals.pdf")))

        tfig, taxes, ffig, faxes = plot_chain(sampler, source=sourceid, stop=stop,
                                              show_trajectories=True, equal_axes=True)
        ffig.savefig(os.path.join('figures', fn.replace(".pkl", "_fluxchain.pdf")))
        tfig.savefig(os.path.join('figures', fn.replace(".pkl", "_shapechain.pdf")))

    if True:
        outname = os.path.join('figures', fn.replace(".pkl", ".png"))
        pinds = scene.sources[0].nparam * sourceid + psub
        label = np.array(scene.sources[sourceid].parameter_names)[psub]
        hmc_movie(sampler, iterations, stamps_to_show, pinds,
                  label=label, outname=outname, png=True)

