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
               show_trajectories=False):
    ndim = sampler.scene.sources[0].nparam
    p0 = sampler.truths
    filters = sampler.filters
    nband = len(filters)

    
    ffig, faxes = pl.subplots(nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios':[2, 1]})
    for j in range(nband):
        i = source * ndim + j
        l, n = [], 0
        if show_trajectories:
            for traj in sampler.trajectories:
                faxes[j, 0].plot(np.arange(len(traj)) + n,
                                traj[:, i], color='r')
                n += len(traj)
                l.append(n)
            cut = l[start]
        else:
            l = np.arange(sampler.chain.shape[0])
            cut = start
        faxes[j, 0].plot(np.array(l), sampler.chain[:, i],'o')
        faxes[j, 0].axhline(p0[i], linestyle=':', color='k')
        faxes[j, 1].hist(sampler.chain[start:, i], alpha=0.5, bins=30, orientation='horizontal')
        faxes[j, 1].axhline(p0[i], linestyle=':', color='k')
        faxes[j, 0].axvline(cut, linestyle='--', color='r')

    faxes = prettify_chains(faxes, filters)
    
    tfig, taxes = pl.subplots(ndim-nband, 2, figsize=(8.5, 9.),
                              gridspec_kw={'width_ratios':[2, 1]})
    for j in range(ndim-nband):
        i = source * ndim + nband + j
        l, n = [], 0
        if show_trajectories:
            for traj in sampler.trajectories:
                taxes[j, 0].plot(np.arange(len(traj)) + n,
                                traj[:, i], color='r')
                n += len(traj)
                l.append(n)
            cut = l[start]
        else:
            l = np.arange(sampler.chain.shape[0])
            cut = start
        taxes[j, 0].plot(np.array(l), sampler.chain[:, i],'o')
        taxes[j, 0].axhline(p0[i], linestyle=':', color='k')
        taxes[j, 1].hist(sampler.chain[start:, i], alpha=0.5, bins=30, orientation='horizontal')
        taxes[j, 1].axhline(p0[i], linestyle=':', color='k')
        taxes[j, 0].axvline(cut, linestyle='--', color='r')

    pnames = ['RA', 'Dec', '$\sqrt{b/a}$', 'PA (rad)', 'n', 'r$_h$']
    taxes = prettify_chains(taxes, pnames)
    

    return tfig, taxes, ffig, faxes


def prettify_chains(axes, labels, fontsize=10):
    [ax.set_ylabel(p) for ax, p in zip(axes[:, 0], labels)]
    [ax.set_xticklabels('') for ax in axes[:-1, 0]]
    [ax.set_xticklabels('') for ax in axes[:, 1]]
    [ax.yaxis.tick_right() for ax in axes[:, 1]]
    [ax.set_ylabel(p) for ax, p in zip(axes[:, 1], labels)]
    [ax.yaxis.set_label_position("right") for ax in axes[:, 1]]
    axes[-1, 0].set_xlabel('HMC iteration')
    
    [item.set_fontsize(fontsize)
     for ax in axes.flat
     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())
    ]
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
