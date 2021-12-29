from itertools import product
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from forcepho.sources import Star, Scene
from forcepho.slow.stamp import PostageStamp
from forcepho.slow.psf import PointSpreadFunction
from forcepho.slow.likelihood import make_image, plan_sources, WorkPlan


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


def get_image(x, y, scene, stamp, **extras):
    theta = np.array([1, x, y])
    im, partials = make_image(scene, stamp, Theta=theta, **extras)
    return im


def rebin(a, new_shape):
    M, N = a.shape
    m, n = new_shape
    if m < M:
        return a.reshape((m, int(M/m), n, int(N/n))).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)


def compare(x, y, source, native, oversampled):
    image = get_image(x, y, source, native)
    oimage = get_image(x, y, source, oversampled)
    rimage = rebin(oimage, image.shape) * oversample**2


def display(ims, labels, radius=None):
    """
    ims : list of ndarrays
        oversampled, rebinned oversampled, direct, residual
    """
    #nix = int(np.ceil(np.sqrt(len(ims) + 1)))
    #niy = int(np.ceil(len(ims) / nix))
    nix, niy = 3, 2
    fig, axes = pl.subplots(nix, niy, figsize=(8.5, 8.5))
    for i, im in enumerate(ims):
        ax = axes.flat[i]
        c = ax.imshow(im.T, origin='lower')
        fig.colorbar(c, ax=ax)
        ax.text(0.1, 0.9, labels[i], color="magenta", transform=ax.transAxes)

    y, ylabel = ims[-1].flatten(), labels[-1]
    if radius is None:
        x, xlabel = ims[1].flatten(), labels[1]
    else:
        x, xlabel = radius.flatten(), "radius (pixels)"

    ax = axes.flat[-1]
    ax.plot(x, y, 'o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_visible(False)

    if radius is None:
        ax.set_xscale("log")
        ax.set_xlim(ims[1].max()*1e-4, ims[1].max()*1.1)

    return fig, axes


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

    pdf = PdfPages('undersample_1storder.pdf')
    xs = [0.0, 0.3, 0.5]
    coordlist = list(product(xs, xs))

    pars = []
    rmax = []

    for fwhm in 1.0, 1.2, 1.4, 1.6, 1.8, 2.0:
        sigma = fwhm / 2.355
        native.psf.covariances = np.array([np.eye(2) * sigma**2])
        oversampled.psf.covariances = np.array([np.eye(2) * (sigma / dx)**2])

        for x, y in coordlist:
            source.ra = x
            source.dec = y
            radius = np.hypot(native.xpix - (native.crpix[0] + source.ra),
                              native.ypix - (native.crpix[1] + source.dec),)

            nimage, _ = source.render(native, compute_deriv=False, second_order=True)
            dimage, _ = source.render(native, compute_deriv=False, second_order=False)
            oimage, _ = source.render(oversampled, compute_deriv=False)
            nimage = nimage.reshape(native.nx, native.ny)
            dimage = dimage.reshape(native.nx, native.ny)
            oimage = oimage.reshape(oversampled.nx, oversampled.ny)
            rimage = rebin(oimage, nimage.shape) * oversample**2
            residual = 100 * (nimage - rimage) / rimage.max()
            r2 = 100 * (dimage - rimage) / rimage.max()
            ims = [oimage, rimage, nimage, residual]
            labels = ['oversampled (by {})'.format(oversample), 'rebinned',
                      'direct (1st order)', 'residual as % of max']

            fig, axes = display(ims, labels, radius=radius)
            axes[-1, 0].set_visible(False)
            axes[-1, 1].axhline(0.0, linestyle=":", color="k")
            title_string = 'FWHM={} pixels,\n$\\Delta x={{{}}}, \\Delta y={{{}}}$\nsums={:.3},{:.3},{:.3}'
            title = title_string.format(fwhm, x, y, oimage.sum(), rimage.sum(), nimage.sum())
            #fig.text(0.14, 0.8, title, transform=fig.transFigure)
            fig.suptitle(title)
            fig.tight_layout()
            pdf.savefig(fig)
            pl.close(fig)

            pars.append((fwhm, x, y))
            delt = residual.flat[np.argmax(np.abs(residual))], r2.flat[np.argmax(np.abs(r2))]
            rmax.append(delt)

    pdf.close()

    pl.ion()
    fig, ax = pl.subplots()
    ax.plot(np.array(pars)[:, 0], np.array(rmax)[:, 0], "o", label="2nd order")
    ax.plot(np.array(pars)[:, 0], np.array(rmax)[:, 1], "o", label="0th order")
    ax.set_xlabel("Gaussian FWHM (pixels)")
    ax.set_ylabel("maximum residual deviation, as % of brightest pixel")
    ax.axhline(0.0, linestyle=":", color="k")
    ax.legend()
    fig.savefig("oversample_1storder_summary.png", dpi=200)