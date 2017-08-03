from itertools import product
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from proto import PostageStamp, Star, PointSpreadFunction
from likelihood import model_image


def get_stamp(n, dx=1):
    stamp = PostageStamp()
    stamp.nx, stamp.ny = n, n
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))
    stamp.distortion = np.eye(2) / dx

    stamp.crval = np.zeros([2])
    stamp.crpix = np.zeros([2])
    stamp.psf = PointSpreadFunction()
    return stamp


def get_image(x, y, source, stamp):
    stamp.residual = np.zeros(stamp.npix)
    theta = np.array([1, x, y, 1, 0, 0, 0])
    residual, partials = model_image([theta], [source], stamp,
                                     use_gradients=use)

    return -residual.reshape(stamp.nx, stamp.ny)


def rebin(a, new_shape):
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)


def compare(x, y, source, native, oversampled):
    
    image = get_image(x, y, source, native)
    oimage = get_image(x, y, source, oversampled)
    rimage = rebin(oimage, image.shape) * oversample**2


if __name__ == "__main__":

    source = Star()
    use = slice(0, 3)
     
    # FWHM in native pixels
    fwhm = 1.0
    sigma = fwhm/2.355

    # native resolution
    native = get_stamp(30)
    native.psf.covariances = np.array([np.eye(2) * sigma**2])
    native.crpix = np.array([15, 15])

    # oversampled image
    oversample = 8
    dx = 1./oversample
    oversampled = get_stamp(int(30 / dx), dx)
    oversampled.psf.covariances = np.array([np.eye(2) * (sigma / dx)**2])
    oversampled.crpix = np.array([oversampled.nx/2, oversampled.ny/2]) + oversample/2. - 0.5

    pdf = PdfPages('undersample.pdf')
    xs = [0.0, 0.25, 0.5]
    coordlist = list(product(xs, xs))
    
    for x, y in coordlist:
        image = get_image(x, y, source, native)
        oimage = get_image(x, y, source, oversampled)
        rimage = rebin(oimage, image.shape) * oversample**2
        ims = [oimage, rimage, image, image/rimage]
        label = ['oversampled (by {})'.format(oversample), 'rebinned',
                 'direct (2nd order)', 'direct/rebinned']
        fig, axes = pl.subplots(3, 2, figsize=(8.5, 8.5))
        for i, im in enumerate(ims):
            ax = axes.flat[i+1]
            c = ax.imshow(im.T, origin='lower')
            fig.colorbar(c, ax=ax)
            ax.text(0.1, 0.9, label[i], transform=ax.transAxes)

        ax = axes.flat[-1]
        ax.plot(rimage.flatten(), (image/rimage).flatten(), 'o')
        ax.set_xlim(1e-3, 0.6)
        ax.set_ylim(0, 1.2)
        ax.axhline(1.0, linestyle=':', color='k')
        ax.set_xscale('log')
        ax.set_xlabel('counts (rebinned)')
        ax.set_ylabel('direct/rebinned')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_visible(False)
        axes[0,0].set_visible(False)
        title = 'FWHM={} pixels,\n$\\Delta x={{{}}}, \\Delta y={{{}}}$'.format(fwhm, x, y)
        fig.text(0.14, 0.8, title,
                 transform=fig.transFigure)
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()

