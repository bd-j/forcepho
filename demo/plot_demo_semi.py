import numpy as np
import matplotlib.pyplot as pl
import pickle, sys

#from demo_semi import setup_scene, Posterior
from forcepho.likelihood import make_image

def load_results(fn="sampler_demo_semi_v1.pkl"):
    with open(fn, "rb") as f:
        sampler = pickle.load(f)

    scene = sampler.scene
    plans = sampler.plans
    return sampler, scene, plans


def plot_data(stamps, filters, offsets):
    dfig, daxes = pl.subplots(len(filters), len(offsets), sharex=True, sharey=True)
    daxes = np.atleast_2d(daxes)
    for i, stamp in enumerate(stamps):
        daxes.flat[i].imshow(stamp.pixel_values.T, origin='lower')

    [ax.set_title("(dx, dy)=\n({}, {})".format(*o), fontsize=12) for (o, ax) in zip(offsets, daxes[0, :])]
    [ax.set_ylabel("{}".format(f)) for (f, ax) in zip(filters, daxes[:, 0])]
    return dfig, daxes


def plot_chain(sampler, start=1000, source=0,
               show_trajectories=False, equal_axes=False):
    ndim = sampler.scene.sources[0].nparam
    p0 = sampler.truths
    filters = sampler.filters
    nband = len(filters)

    ffig, faxes = pl.subplots(nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios':[2, 1]})
    for j in range(nband):
        i = source * ndim + j
        _ = one_chain(faxes[j, :], sampler, i, p0[i], start,
                      show_trajectories=show_trajectories)

    faxes = prettify_chains(faxes, filters, equal_axes=equal_axes,
                            show_trajectories=show_trajectories)
    
    tfig, taxes = pl.subplots(ndim-nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios':[2, 1]})
    for j in range(ndim-nband):
        i = source * ndim + nband + j
        _ = one_chain(taxes[j, :], sampler, i, p0[i], start,
                      show_trajectories=show_trajectories)

    pnames = ['RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)', 'n', 'r$_h$']
    taxes = prettify_chains(taxes, pnames, equal_axes=equal_axes,
                            show_trajectories=show_trajectories)

    return tfig, taxes, ffig, faxes


def one_chain(axes, sampler, i, truth=None, start=0, show_trajectories=False):
    l, n = [], 0
    if show_trajectories:
        label = "trajectories"
        for traj in sampler.trajectories:
            axes[0].plot(np.arange(len(traj)) + n,
                         traj[:, i], color='r', label=label)
            n += len(traj)
            l.append(n)
            label=None
        cut = l[start]
    else:
        l = np.arange(sampler.chain.shape[0])
        cut = start
    axes[0].plot(np.array(l), sampler.chain[:, i],'o', label="samples")
    axes[1].hist(sampler.chain[start:, i], alpha=0.5, bins=30, orientation='horizontal')
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
        axes[0, 0].legend()
    else:
        axes[-1, 0].set_xlabel('HMC iteration')
    
    [item.set_fontsize(fontsize)
     for ax in axes.flat
     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())
    ]

    if equal_axes:
        [ax[1].set_ylim(ax[0].get_ylim()) for ax in axes]

    
    return axes

    
def plot_model_images(pos, scene, stamps):
    vals = pos
    rfig, raxes = pl.subplots(len(stamps), 3, figsize=(12, 9),
                              sharex=True, sharey=True)
    raxes = np.atleast_2d(raxes)
    for i, stamp in enumerate(stamps):
        im, grad = make_image(scene, stamp, Theta=vals)
        raxes[i, 0].imshow(stamp.pixel_values.T, origin='lower')
        raxes[i, 1].imshow(im.T, origin='lower')
        resid = raxes[i, 2].imshow((stamp.pixel_values - im).T, origin='lower')
        rfig.colorbar(resid, ax=raxes[i,:].tolist())
        text = "{}\n({}, {})".format(stamp.filtername, stamp.crval[0], stamp.crval[1])
        ax = raxes[i, 1]
        ax.text(0.6, 0.1, text, transform=ax.transAxes, fontsize=10)

    labels = ['Data', 'Model', 'Data-Model']
    [ax.set_title(labels[i]) for i, ax in enumerate(raxes[0,:])]
    return rfig, raxes


def plot_seds(sampler, start=1000):
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
        samples = sampler.chain[start:, s:s+nband]
        pct = np.percentile(samples, [16, 50, 84], axis=0)
        neg = pct[1, :] - pct[0,:]
        plus = pct[2, :] - pct[1, :]
        print("{}: {}".format(isource, (neg+plus) / pct[1,:]))
        ax.errorbar(wave, pct[1,:], yerr=[neg, plus], label="Posterior",
                    capsize=5.0, capthick=2.0)
        ax.plot(wave, fluxes, '-o', label="Truth".format(isource))

    [ax.set_xlim(0.8, 2.9) for ax in saxes.flat]
    [ax.set_ylabel("Flux") for ax in saxes.flat]
    saxes.flat[-1].set_xlabel("$\lambda (\mu m)$")
    return sfig, saxes


if __name__ == "__main__":
    sampler, scene, plans = load_results(fn="semi_results_snr10.pkl")
    stamps = [p.stamp for p in plans]
    stamps_to_show = [stamps[0], stamps[1], stamps[13], stamps[22]]
    pos = sampler.chain[-1, :]


    dfig, daxes = plot_data(stamps, sampler.filters, sampler.offsets)

    sfig, saxes = plot_seds(sampler)
    
    rfig, raxes = plot_model_images(pos, scene, stamps_to_show)

    tfig, taxes, ffig, faxes = plot_chain(sampler, source=2)
