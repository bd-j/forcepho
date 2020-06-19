#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""make_test_image.py - Use galsim to generate some test images that can be
used in regression testing. These will be single band images.  The images
include
   * galaxy_grid - A small grid of galaxies with varied shape parameters
                   (size, sersic, axis ratio) and at different S/N
   * shift_grid - The same galaxy (with the same rng) at different sub-pixel positions.
"""

import sys
from argparse import ArgumentParser
from itertools import product

import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from astropy.wcs import WCS
import galsim


grid_points = dict(rhalf=np.arange(0.05, 0.3, 0.1),
                   sersic=np.arange(1.5, 5.0, 1.0),
                   q=np.array([0.5, 0.95]),
                   snr=np.array([3, 10, 30, 100]))


def make_catalog(grid_points, band="fclear", pixel_scale=0.03, noise=1.0):
    grid_keys = list(grid_points.keys())
    cols = np.unique(grid_keys + ["pa", "ra", "dec", "id", "x", "y"] + [band])
    cat_dtype = np.dtype([(c, np.float64) for c in cols])

    grid = np.array(list(product(*[grid_points[k] for k in grid_keys])))
    n_gal = len(grid)
    cat = np.zeros(n_gal, dtype=cat_dtype)

    for i, k in enumerate(grid_keys):
        cat[k] = grid[:, i]

    n_pix = np.pi * (cat["rhalf"] / pixel_scale)**2
    cat[band] = 2 * cat["snr"] * np.sqrt(n_pix) * noise
    return cat


def make_image(cat, n_pix_per_side, n_pix_per_gal,
               pixel_scale=0.03, sigma_psf=1.):

    psf = galsim.Gaussian(flux=1., sigma=sigma_psf)
    image = galsim.ImageF(n_pix_per_side, n_pix_per_side, scale=pixel_scale)

    for i, row in enumerate(cat):
        gal = galsim.Sersic(half_light_radius=row["rhalf"],
                            n=row["sersic"], flux=row[args.BAND])
        egal = gal.shear(q=row["q"], beta=row["pa"] * galsim.radians)
        final_gal = galsim.Convolve([psf, egal])

        # place the galaxy and draw it
        x, y = row["x"] + 1, row["y"] + 1
        bounds = galsim.BoundsI(x - 0.5*n_pix_per_gal + 1, x + 0.5*n_pix_per_gal - 1,
                                y - 0.5*n_pix_per_gal + 1, y + 0.5*n_pix_per_gal - 1)
        final_gal.drawImage(image[bounds], add_to_image=True)

    return image


def make_header(im, args):

    pixel_scale = args.PIXEL_SCALE

    header = {}
    header["CRPIX1"] = 0.0
    header["CRPIX2"] = 0.0
    header["CRVAL1"] = 53.187
    header["CRVAL2"] = -27
    header["CD1_1"] = -pixel_scale / 3600.
    header["CD2_2"] = pixel_scale / 3600.
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["FILTER"] = args.BAND
    header["NOISE"] = args.NOISE
    header["PSFSIGMA"] = args.PSF_SIGMA
    header["PIXSCALE"] = pixel_scale

    wcs = WCS(header)

    return header, wcs


def write_im(name, array, header, **kwargs):
    # note the transpose to get numpy style!
    with fits.HDUList(fits.PrimaryHDU(array)) as hdul:
        hdul[0].header.update(header)
        hdul[0].header.update(**kwargs)
        hdul.writeto(name, overwrite=True)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str,
                        default="galsim_galaxy_grid")
    parser.add_argument("--PIXEL_SCALE", type=float,
                        default=0.03)
    parser.add_argument("--NOISE", type=float,
                        default=1.0)
    parser.add_argument("--BAND", type=str,
                        default="Fclear")
    parser.add_argument("--PSF_SIGMA", type=float,
                        default=-99)
    args = parser.parse_args()
    if args.PSF_SIGMA < 0:
        args.PSF_SIGMA = 2 * args.PIXEL_SCALE / 2.355

    origin = 0
    nhalf = 5

    cat = make_catalog(grid_points, band=args.BAND, noise=args.NOISE,
                       pixel_scale=args.PIXEL_SCALE)

    n_gal = len(cat)
    n_gal_per_side = int(np.ceil(np.sqrt(n_gal)))
    n_pix_per_gal = int(np.ceil(nhalf * np.max(cat["rhalf"] / args.PIXEL_SCALE) * 2))
    ix = np.floor(np.arange(n_gal) / n_gal_per_side).astype(int)
    iy = np.mod(np.arange(n_gal), n_gal_per_side)
    cat["x"] = (ix + 0.5) * n_pix_per_gal
    cat["y"] = (iy + 0.5) * n_pix_per_gal

    n_pix_per_side = n_pix_per_gal * n_gal_per_side
    n_pix_per_side += n_pix_per_side % 8

    im = make_image(cat, n_pix_per_side, n_pix_per_gal,
                    sigma_psf=args.PSF_SIGMA, pixel_scale=args.PIXEL_SCALE)
    truth = im.copy()
    im.addNoise(galsim.GaussianNoise(sigma=args.NOISE))
    noisy = im.copy()

    header, wcs = make_header(im, args)
    ra, dec = wcs.all_pix2world(cat["x"], cat["y"], origin)
    cat["ra"] = ra
    cat["dec"] = dec

    pl.ion()
    fig, ax = pl.subplots()
    ax.imshow(im.array, origin="lower")

    sys.exit()

    write_im("data/{}_truth.fits".format(args.name), truth.array, header, NOISE=0)
    write_im("data/{}_noisy.fits".format(args.name), noisy.array, header)
    write_im("data/{}_unc.fits".format(args.name), args.NOISE*np.ones_like(truth.array), header)
    with fits.HDUList([]) as hdul:
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.BinTableHDU(cat))
        hdul[0].header["FILTERS"] = ",".join([args.BAND])
        hdul[1].header["FILTERS"] = ",".join([args.BAND])
        hdul.writeto("data/{}_cat.fits".format(args.name), overwrite=True)
