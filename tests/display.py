#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, time, os, sys
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits

sys.path.append(os.path.abspath("./verification"))
from make_reference import make_stamp, config_to_cat, forcepho_slow_model
sys.path.append(os.path.abspath("./galsim_tests"))
from make_galsim_image import galsim_model


def show(fim, gim, figsize=(10.6, 4.2)):

    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import get_cmap
    cmap = get_cmap('viridis')

    fig = pl.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[1, 15], hspace=0.01, top=0.8)
    ax = fig.add_subplot(gs[1, 0])
    axes = [ax] + [fig.add_subplot(gs[1, i], sharex=ax, sharey=ax) for i in range(1, 3)]
    cbars = [fig.add_subplot(gs[0, i]) for i in range(3)]

    x, y = np.log10(fim.T), np.log10(gim.T)
    #x, y = fim.T, gim.T
    vmin, vmax = np.nanmin(x), np.nanmax(x)
    cb = axes[0].imshow(x, origin="lower", vmin=vmin, vmax=vmax)
    pl.colorbar(cb, cax=cbars[0], orientation="horizontal",
                label="log(model)")

    cb = axes[1].imshow(y, origin="lower", vmin=vmin, vmax=vmax)
    pl.colorbar(cb, cax=cbars[1], orientation="horizontal",
                label="log(true)")

    rpct = (fim-gim).T/gim.max()*100
    cb = axes[2].imshow(rpct, origin="lower", vmin=-3, vmax=3)
    pl.colorbar(cb, cax=cbars[2], orientation="horizontal",
                label="(model-true)/max (%)")

    return fig, axes, rpct


def comparison_simplepsf(config, sigma=0.5, scale=0.03):
    """Test the sersic mixture, with a single gaussian PSF applied to both

    sigma : float
        dispersion of PSF in pixels

    scale : float
        arcsec per pixel
    """
    import galsim
    from forcepho.slow.psf import PointSpreadFunction

    stamp = make_stamp(config.band, scale=scale, nx=config.nx, ny=config.ny)
    cat, hdr, wcs = config_to_cat(config, stamp, origin=1)

    fpsf = PointSpreadFunction()
    fpsf.covariances *= sigma**2
    gpsf = galsim.Gaussian(flux=1., sigma=sigma*scale)

    fim, _, fgal = forcepho_slow_model(cat[0], stamp, config.band, psf=fpsf)
    gim, ggal = galsim_model(cat[0], stamp, config.band, psf=gpsf)

    pl.ion()
    fig, axes, resid = show(fim, gim)

    return fim, gim, resid, fig


def comparison_fullpsf(config, psffile="", scale=0.03):
    """Test the sersic mixture, with an actual realistic JWST PSF and GMM

    psffile : string
        dispersion of PSF in pixels

    scale : float
        arcsec per pixel
    """
    import galsim
    from forcepho.slow.psf import PointSpreadFunction

    stamp = make_stamp(config.band, scale=scale, nx=config.nx, ny=config.ny)
    cat, hdr, wcs = config_to_cat(config, stamp, origin=1)

    psfim = fits.getdata(psffile, 1)
    psfhdr = fits.getheader(psffile, 0)
    psfcat = fits.getdata(psffile, -1)

    fpsf = PointSpreadFunction(parameters=psfcat)
    pim = galsim.Image(np.ascontiguousarray(psfim.astype(np.float64)), scale=scale / psfhdr["DET_SAMP"])
    gpsf = galsim.InterpolatedImage(pim)

    fim, _, fgal = forcepho_slow_model(cat[0], stamp, config.band, psf=fpsf)
    gim, ggal = galsim_model(cat[0], stamp, config.band, psf=gpsf)

    pl.ion()
    fig, axes, resid = show(fim, gim)

    return fim, gim, resid, fig


if __name__ == "__main__":

    sw = ["F090W", "F115W", "F150W", "F200W"]
    lw = ["F277W", "F335M", "F356W", "F410M", "F444W"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--psffile", type=str, default="./data/psfs/psf_jwst_oct21_ng4m0_{band}.fits")
    parser.add_argument("--sersic", type=float, default=2.2, help="value of n, between 1 and 5")
    parser.add_argument("--rhalf", type=float, default=0.1, help="in arcsec")
    parser.add_argument("--flux", type=float, default=1.0, help="total object flux")
    parser.add_argument("--nx", type=int, default=64, help="If even, source is centered at pixel x edge")
    parser.add_argument("--ny", type=int, default=64, help="If even, source is centered at pixel y edge")
    parser.add_argument("--band", type=str, default="F200W", choices=sw + lw)
    args = parser.parse_args()
    args.psffile = args.psffile.format(band=args.band.lower())

    ts = time.strftime("%Y%b%d", time.localtime())
    args.outname = (f"../data/images/reference/reference-{ts}_{args.band.lower()}_"
                    f"sersic={args.sersic:.1f}_rhalf={args.rhalf:.2f}.fits")

    if args.band.upper() in sw:
        scale = 0.03
    elif args.band.upper() in lw:
        scale = 0.06


    # Do som simple comparisons
    fim1, gim1, resid1, fig1 = comparison_simplepsf(args, sigma=0.5, scale=scale)
    diff = resid1.flatten()[np.argmax(np.abs(resid1))]
    fig1.suptitle(f'Test1: n={args.sersic:.1f}, re={args.rhalf:0.2f}"\nmax diff= {diff:+.2f}% of max pixel')

    fim2, gim2, resid2, fig2 = comparison_fullpsf(args, psffile=args.psffile, scale=scale)
    diff = resid2.flatten()[np.argmax(np.abs(resid2))]
    fig2.suptitle(f'Test2 ({args.band}): n={args.sersic:.1f}, re={args.rhalf:0.2f}"\nmax diff= {diff:+.2f}% of max pixel')