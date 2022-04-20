from itertools import product
import numpy as np
import matplotlib.pyplot as pl

from forcepho.sources import Star, Scene
from forcepho.slow.stamp import PostageStamp
from forcepho.slow.psf import PointSpreadFunction


def get_stamp(n, dx=1, filtername="band"):
    stamp = PostageStamp()
    stamp.filtername = filtername
    stamp.nx, stamp.ny = n, n
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))
    stamp.scale = np.eye(2) / dx
    stamp.dpix_dsky = np.eye(2) / dx

    stamp.crval = np.zeros([2])
    stamp.crpix = np.zeros([2])
    stamp.psf = PointSpreadFunction()
    return stamp


def rebin(a, new_shape):
    M, N = a.shape
    m, n = new_shape
    if m < M:
        return a.reshape((m, int(M/m), n, int(N/n))).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)


if __name__ == "__main__":

    source = Star()
    source.flux = 1.0
    scene = Scene([source])
    width = 10

    # native resolution
    native = get_stamp(width)
    native.crpix = np.array([width/2., width/2.])

    # oversampled image
    oversample = 8
    dx = 1.0 / oversample
    oversampled = get_stamp(int(width / dx), dx)
    oversampled.crpix = np.array([oversampled.nx/2, oversampled.ny/2]) + oversample/2. - 0.5

    xs = [0.0, 0.3, 0.5]
    coordlist = list(product(xs, xs))

    pars, rmax, tot = [], [], []

    for fwhm in 1.0, 1.2, 1.35, 1.5, 1.8, 2.0, 2.5, 3.0:
        sigma = fwhm / 2.355
        native.psf.covariances = np.array([np.eye(2) * sigma**2])
        oversampled.psf.covariances = np.array([np.eye(2) * (sigma / dx)**2])

        for x, y in coordlist:
            source.ra = x
            source.dec = y
            radius = np.hypot(native.xpix - (native.crpix[0] + source.ra),
                              native.ypix - (native.crpix[1] + source.dec),)

            # second order image
            nimage, _ = source.render(native, compute_deriv=False, second_order=True)
            nimage = nimage.reshape(native.nx, native.ny)
            # zeroth order image
            dimage, _ = source.render(native, compute_deriv=False, second_order=False)
            dimage = dimage.reshape(native.nx, native.ny)
            # oversampled image
            oimage, _ = source.render(oversampled, compute_deriv=False)
            oimage = oimage.reshape(oversampled.nx, oversampled.ny)
            # rebinned oversampled image
            timage = rebin(oimage, nimage.shape) * oversample**2
            # residual image
            nresidual = 100 * (nimage - timage) / timage.max()
            dresidual = 100 * (dimage - timage) / timage.max()

            pars.append((fwhm, x, y))
            # Max deltas
            delt = nresidual.flat[np.argmax(np.abs(nresidual))], dresidual.flat[np.argmax(np.abs(dresidual))]
            rmax.append(delt)
            # totals
            rflux = nimage.sum() / timage.sum(), dimage.sum() / timage.sum()
            tot.append(rflux)

            if (fwhm == 1.5) & (x == 0) & (y == 0):
                images = dict(oversampled=oimage.copy(),
                              truth=timage.copy(),
                              second_order=nimage.copy(),
                              zeroth_order=dimage.copy(),
                              second_order_residual=nresidual.copy(),
                              zeroth_order_residual=dresidual.copy())
    pars = np.array(pars)
    rmax = np.array(rmax)
    tot = np.array(tot)

    # set up axes
    pl.ion()
    ncol = 3
    fig = pl.figure(figsize=(9, 11.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, ncol, height_ratios=[1, 15, 15, 1, 25],
                  hspace=0.2, wspace=0.15,
                  left=0.08, right=0.9, bottom=0.08, top=0.93)
    cbaxes = np.array([[fig.add_subplot(gs[0, i]) for i in range(ncol)],
                       [fig.add_subplot(gs[3, i]) for i in range(ncol)]])
    imaxes = np.array([[fig.add_subplot(gs[1, i]) for i in range(ncol)],
                       [fig.add_subplot(gs[2, i]) for i in range(ncol)]])
    rax = fig.add_subplot(gs[-1, :])

    # plot the images
    show = ["truth", "zeroth_order", "second_order",
            "oversampled", "zeroth_order_residual", "second_order_residual"]
    for i, ax in enumerate(imaxes.flat):
        label = show[i]
        im = images[label]
        if i < 3:
            kw = dict(vmin=0, vmax=0.35)
        elif i > 3:
            kw = dict(vmin=-5, vmax=25)
        else:
            kw = {}
        c = ax.imshow(im.T, origin='lower', **kw)
        pl.colorbar(c, cax=cbaxes.flat[i], orientation="horizontal")
        ax.text(0.1, 0.9, label.replace("_", " "), color="magenta", transform=ax.transAxes)

    # plot the residual vs fwhm
    rax.plot(pars[:, 0], rmax[:, 0], "o", label="2nd order")
    rax.plot(pars[:, 0], rmax[:, 1], "o", label="0th order")
    rax.set_xlabel("Gaussian FWHM (pixels)")
    rax.set_ylabel("maximum residual deviation, as % of brightest pixel")
    rax.axhline(0.0, linestyle=":", color="k")
    rax.legend()
    fig.savefig("figures/undersample.png", dpi=400)
