# -*- coding: utf-8 -*-

from itertools import product
import os, argparse
import numpy as np

import galsim
from astropy.io import fits


__all__ = ["make_stamp",
           "config_to_cat", "get_grid_params",
           "galsim_model", "get_galsim_psf",
           "make_galsim_image"]


swbands = ["f070w", "f090w", "f115w", "f150w", "f200w"]
lwbands = ["f277w", "f335m", "f356w", "f410m", "f444w"]


def make_stamp(band, scale=0.03, nx=64, ny=64):
    """Make a simple stamp instance
    """
    from forcepho.slow.stamp import PostageStamp
    stamp = PostageStamp(nx, ny)
    stamp.scale = np.eye(2) / scale
    stamp.crval = [53.0, -27.0]
    stamp.filtername = band
    return stamp


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
    from forcepho.superscene import sourcecat_dtype

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


def get_grid_params(config, taskID=None):

    import yaml
    if taskID is None:
        taskID = config.parameters_index

    if getattr(config, "sersic", None):
        params = dict(band=config.band, rhalf=config.rhalf,
                      sersic=config.sersic, snr=config.snr)

    elif "yml" in config.test_grid:
        with open(config.test_grid, "r") as f:
            grid = yaml.load(f, Loader=yaml.Loader)
        names = list(grid.keys())
        names.sort()
        dtype = np.dtype([(k, np.array(grid[k]).dtype) for k in names])
        params = list(product(*[grid[k] for k in names]))
        partable = np.array(params, dtype=dtype)
        params = dict(zip(names, params[taskID]))

        fits.writeto(config.test_grid.replace("yml", "fits"),
                     partable, overwrite=True)

    elif "fits" in config.test_grid:
        partable = fits.getdata(config.test_grid)
        params = partable[taskID]
        params = dict(zip(partable.dtype.names, params))

    return params


def galsim_model(catrow, stamp, band, psf=None):
    """
    Parameters
    ----------
    catrow : structured ndarray
        Source parameters as a forcepho standard catalog format

    stamp : forcepho.slow.stamp.PostageStamp() instance.
        The image parameters to use for the model

    band : string
        The band name, used to index the source catalog.

    psf : optional
        A GalSim object that represent the PSF

    Returns
    -------
    model : ndarray of shape (nx, ny)
        The GalSim model image, after convolution with the PSF

    gal : GalSim object
        A GSobject representing the convolved galaxy

    """
    import galsim

    pixel_scale = 1 / np.sqrt(np.linalg.det(stamp.scale))
    image = galsim.ImageF(stamp.nx, stamp.ny, scale=pixel_scale)
    gal = galsim.Sersic(half_light_radius=catrow["rhalf"],
                        n=catrow["sersic"], flux=catrow[band])
    if catrow["q"] != 1:
        # in forcepho q = sqrt(b/a), and PA is in radians
        gal = gal.shear(q=catrow["q"]**2, beta=catrow["pa"] * galsim.radians)

    if psf is not None:
        final_gal = galsim.Convolve([psf, gal])
    else:
        final_gal = gal

    final_gal.drawImage(image, add_to_image=True)

    model = image.array.T

    return model, gal


def get_galsim_psf(sigma_psf, config, det_samp=4):
    """
    Parameters
    ----------
    sigma_psf : float, arcsec

    config : namespace
        Must have:
        * psf_type - 'simple' | 'mixture' | 'webbpsf'
        * psfstore - image
        * bandname - string, e.g. 'F277W'
        * scale - float, arcsec per science detector pxel

    det_samp : int
        Number of PSF pixels per science detector pixel.

    """
    if config.psf_type == "simple":
        gpsf = galsim.Gaussian(flux=1., sigma=sigma_psf)
    elif config.psf_type == "mixture":
        raise NotImplementedError
    elif config.psf_type == "webbpsf":
        import h5py
        b = config.band.upper()
        with h5py.File(config.psfstore, "r") as pdat:
            psfarr = pdat[b]["truth"][:]  # 1-d array
            nx, ny = pdat[b].attrs["nx"], pdat[b].attrs["ny"]
            psfim = psfarr.reshape(nx, ny).T  # as [y,x]
        pim = galsim.Image(np.ascontiguousarray(psfim.astype(np.float64)),
                           scale=config.scale/det_samp)
        gpsf = galsim.InterpolatedImage(pim)

    return gpsf


def make_galsim_image(band="", rhalf=0.1, sersic=2.0, snr=10, q=1.0,
                      nx=64, ny=64, flux=1, fwhm_psf=None,
                      psf_type="", psffile="", outname=""):

    config = argparse.Namespace()
    config.band = band
    config.rhalf = rhalf
    config.sersic = sersic
    config.q = q
    config.snr = snr
    config.flux = flux
    config.nx = nx
    config.ny = ny
    config.psf_type = psf_type
    config.psfstore = psffile
    config.outname = outname

    config.scale = 0.03 * (1 + int(config.band.lower() in lwbands))
    if not config.outname:
        config.outname = (f"./data/images/galsim-{config.band}-{config.psf_type}-snr={config.snr:.0f}-"
                          f"sersic={config.sersic:.1f}-rhalf={config.rhalf:.2f}-q={config.q:.2f}.fits")

    stamp = make_stamp(config, nx=config.nx, ny=config.ny, scale=config.scale)
    cat, hdr, wcs = config_to_cat(config, stamp)

    if (config.psf_type == "simple") and fwhm_psf:
        sigma_psf = fwhm_psf / 2.355
    else:
        sigma_psf = config.scale
        fwhm_psf = 2.355 * sigma_psf

    gpsf = get_galsim_psf(sigma_psf, config)
    gim, ggal = galsim_model(cat[0], stamp, config.band, psf=gpsf)

    # --- compute uncertainty ---
    npix = np.pi*(config.rhalf / config.scale)**2
    signal = config.flux / 2
    noise = signal / config.snr
    noise_per_pix = noise / np.sqrt(npix)
    print(f"Noise per pixel = {noise_per_pix}")

    # --- update the header ---
    for k, v in vars(config).items():
        hdr[k] = v
    hdr["FILTER"] = config.band
    hdr["PSFSIG"] = sigma_psf

    # --- write the test image ---
    image = fits.PrimaryHDU(gim.T, header=hdr)
    uncertainty = fits.ImageHDU(np.ones_like(gim.T)*noise_per_pix, header=hdr)
    catalog = fits.BinTableHDU(cat)
    catalog.header["FILTERS"] = ",".join([config.band])

    os.makedirs(os.path.dirname(config.outname), exist_ok=True)
    hdul = fits.HDUList([image, uncertainty, catalog])
    hdul.writeto(config.outname, overwrite=True)
    hdul.close()

    return config.outname


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=str, default="F277W")
    parser.add_argument("--psfstore", type=str, default="./data/stores/psf_jwst_oct21_ng4m0.h5")
    parser.add_argument("--psf_type", type=str, default="webbpsf")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--flux", type=float, default=1.0)
    parser.add_argument("--sersic", type=float, default=2.0)
    parser.add_argument("--rhalf", type=float, default=0.1)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--snr", type=float, default=10)
    parser.add_argument("--outname", type=str, default=None)
    config = parser.parse_args()

    out = make_galsim_image(config.band, config.rhalf, config.sersic, config.snr, config.q,
                            nx=config.nx, ny=config.ny, flux=config.flux,
                            psf_type=config.psf_type, psffile=config.psfstore,
                            outname=config.outname)

    print(f"wrote galsim image to {out}")