#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make a reference forcepho galaxy model using the slow, pythonic code and a
particular JWST PSF approximation.
"""

import argparse, time
import numpy as np

import h5py
from astropy.io import fits
from astropy.wcs import WCS

import forcepho
from forcepho.slow.stamp import PostageStamp, scale_at_sky
from forcepho.slow.psf import PointSpreadFunction
from forcepho.sources import Galaxy
from forcepho.superscene import sourcecat_dtype


def get_githash():
    import os, subprocess
    cwd = os.getcwd()
    os.chdir(__file__)
    process = subprocess.Popen(
        ['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE,
        universal_newlines=True, encoding="utf-8")
    git_head_hash = process.communicate()[0].strip()
    os.chdir(cwd)
    return git_head_hash


def forcepho_slow_model(catrow, stamp, band, psf=None,
                        splinedata="./data/sersic_mog_model.smooth=0.0150.h5"):

    galaxy = Galaxy(splinedata=splinedata)
    galaxy.from_catalog_row(catrow, filternames=[band])

    if psf is None:
        psf = PointSpreadFunction()
    stamp.psf = psf

    im, grad = stamp.render(galaxy)
    im = im.reshape(stamp.nx, stamp.ny)

    return im, grad, galaxy


def config_to_cat(config, stamp, origin=0):
    """Convert a configuration namespace to a structured ndarray in the forcepho
    catalog format.

    Parameters
    ----------
    config : Namespace instance
        must have the attributes `bvand`, `rhalf` and `sersic`, optionally `flux`

    stamp : forcepho.slow.stamp.PostageStamp instance
        Defines the WCS.  Note that the output RA and Dec assume the source to
        be centered in the stamp.

    Returns
    -------
    cat : structured ndarray of shape (1,)
        The source parameters as a catalog row

    hdr : a FITS style header for the input stamp

    wcs : the WCS for the stamp
    """
    cdtype = sourcecat_dtype(bands=[config.band])
    cat = np.zeros(1, dtype=cdtype)
    cat["rhalf"] = config.rhalf
    cat["sersic"] = config.sersic
    cat[config.band] = getattr(config, "flux", 1.0)
    cat["q"] = getattr(config, "q", 1.0)
    cat["pa"] = getattr(config, "pa", 0.0)

    hdul, wcs = stamp.to_fits()
    ra, dec = wcs.all_pix2world((stamp.nx-1) / 2.,
                                (stamp.ny-1) / 2.,
                                origin, ra_dec_order=True)
    cat["ra"] = ra
    cat["dec"] = dec

    return cat, hdul[0].header, wcs


def make_stamp(band, scale=0.03, nx=64, ny=64):
    """Make a stamp instance
    """
    stamp = PostageStamp(nx, ny)
    stamp.scale = np.eye(2) / scale
    stamp.crval = [53.0, -27.0]
    stamp.filtername = band
    return stamp


if __name__ == "__main__":

    sw = ["F090W", "F115W", "F150W", "F200W"]
    lw = ["F277W", "F335M", "F356W", "F410M", "F444W"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--psffile", type=str, default="./data/psf_jwst_oct21_ng4m0.h5")
    parser.add_argument("--splinedata", type=str, default="./data/sersic_mog_model.smooth=0.0150.h5")
    parser.add_argument("--sersic", type=float, default=2.2, help="value of n, between 1 and 5")
    parser.add_argument("--rhalf", type=float, default=0.1, help="in arcsec")
    parser.add_argument("--flux", type=float, default=1.0, help="total object flux")
    parser.add_argument("--nx", type=int, default=64, help="If even, source is centered at pixel x edge")
    parser.add_argument("--ny", type=int, default=64, help="If even, source is centered at pixel y edge")
    parser.add_argument("--band", type=str, default="F200W", choices=sw + lw)
    args = parser.parse_args()

    ts = time.strftime("%Y%b%d", time.localtime())
    args.outname = (f"./data/reference-{ts}_{args.band.lower()}_"
                    f"sersic={args.sersic:.1f}_rhalf={args.rhalf:.2f}.fits")

    if args.band.upper() in sw:
        scale = 0.03
    elif args.band.upper() in lw:
        scale = 0.06

    # --- build the reference image ---
    with h5py.File(args.psffile, "r") as pdat:
        psfcat = pdat[band.upper()]["parameters"][:]
        psfcat = psfcat[0, psfcat["sersic_bin"][0, :] == 0]
    fpsf = PointSpreadFunction(parameters=psfcat)
    assert fpsf.n_gauss == len(psfcat)
    stamp = make_stamp(args.band.upper(), scale=scale, nx=args.nx, ny=args.ny)
    cat, hdr, wcs = config_to_cat(args, stamp, origin=0)
    im, grad, gal = forcepho_slow_model(cat[0], stamp, args.band, psf=fpsf)

    # --- add meta information ---
    for k, v in vars(args).items():
        hdr[k] = v
    hdr["FILTER"] = args.band.upper()
    hdr["FPHOVER"] = forcepho.__version__
    try:
        hdr["TESTVER"] = get_githash()  # in case this is run outside the forcepho repo....
    except:
        pass

    # --- write the reference image ---
    hdr["BUNIT"] = "FLUX"
    hdr["DATE"] = time.strftime("%Y.%m%d-%H.%M", time.localtime())
    image = fits.PrimaryHDU(im.T, header=hdr)
    hdul = [image]
    for g, par in zip(grad, gal.parameter_names):
        hdr["BUNIT"] = f"dFLUX/d{par}"
        hdul.append(fits.ImageHDU(g.reshape(stamp.nx, stamp.ny).T, header=hdr))
    hdr.pop("BUNIT")
    catalog = fits.BinTableHDU(cat)
    catalog.header["FILTERS"] = ",".join([args.band.upper()])
    catalog.header["SHAPCOL"] = ",".join(Galaxy.SHAPE_COLS)
    hdul.append(catalog)

    hdul = fits.HDUList(hdul)
    hdul.writeto(args.outname, overwrite=True)
    print(f"wrote reference image to {args.outname}")