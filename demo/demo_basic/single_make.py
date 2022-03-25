# -*- coding: utf-8 -*-

import sys
from itertools import product
import os, argparse
import numpy as np
import h5py

import galsim
from astropy.io import fits


__all__ = ["make_stamp", "make_scene",
           "make_galsim_image"]


def make_stamp(band, scale=0.03, nx=64, ny=64):
    """Make a simple stamp instance
    """
    from forcepho.slow.stamp import PostageStamp
    stamp = PostageStamp(nx, ny)
    stamp.scale = np.eye(2) / scale
    stamp.crval = [53.0, -27.0]
    stamp.filtername = band
    return stamp


def make_scene(stamp, rhalf=0.15, sersic=1, dist_frac=1.0, flux=1.0, origin=0):
    """Convert a configuration namespace to a structured ndarray in the forcepho
    catalog format.

    Parameters
    ----------
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

    nsource = 1
    band = stamp.filtername

    from forcepho.superscene import sourcecat_dtype
    cdtype = sourcecat_dtype(bands=[band])
    cat = np.zeros(nsource, dtype=cdtype)
    cat["id"] = np.arange(2)
    cat["rhalf"] = np.array(rhalf)
    cat["sersic"] = np.array(sersic)
    cat[band] = np.array(flux)
    cat["q"] = 0.9
    cat["pa"] = np.pi / 2

    hdul, wcs = stamp.to_fits()
    ra, dec = wcs.all_pix2world((stamp.nx-1) / 2.,
                                (stamp.ny-1) / 2.,
                                origin, ra_dec_order=True)
    cat["ra"] = ra
    cat["dec"] = dec + np.arange(nsource) * (np.array(rhalf) * dist_frac) / 3600

    return cat


def galsim_model(scene, stamp, psf=None):
    """
    Parameters
    ----------
    scene : structured ndarray
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
    """
    import galsim

    band = stamp.filtername
    pixel_scale = 1 / np.sqrt(np.linalg.det(stamp.scale))
    image = galsim.ImageF(stamp.nx, stamp.ny, scale=pixel_scale)
    #xcen, ycen = stamp.wcs.all_world2pix(scene[0]["ra"], scene[0]["dec"], 0)

    for catrow in scene:
        gal = galsim.Sersic(half_light_radius=catrow["rhalf"],
                            n=catrow["sersic"], flux=catrow[band])
        # shift the galaxy
        dx = 3600. * (catrow["ra"] - scene[0]["ra"]) * np.cos(np.deg2rad(catrow["dec"]))
        dy = 3600. * (catrow["dec"] - scene[0]["dec"])
        if np.hypot(dx, dy) / pixel_scale > 1e-2:
            print(f"applying shift of {dx}, {dy} arcsec")
            offset = dx / pixel_scale, dy / pixel_scale
        else:
            offset = 0, 0
        # shear the galaxy
        # in forcepho q = sqrt(b/a), and PA is in radians
        if catrow["q"] != 1:
            gal = gal.shear(q=catrow["q"]**2, beta=catrow["pa"] * galsim.radians)

        if psf is not None:
            gal = galsim.Convolve([psf, gal])

        gal.drawImage(image, offset=offset, add_to_image=True)

    return image.array.T


def get_galsim_psf(scale, psf_type="simple", sigma_psf=1.0,
                   bandname=None, psfstore=None):
    """
    Parameters
    ----------
    sigma_psf : float
        pixels
    scale : float
        arcsec per science detector pxel
    psf_type : string
        'simple' | 'mixture' | 'webbpsf'
    psfstore : string
        patch to fits image (WebbPSF) or h5store (mixture)
    bandname : string
        e.g. 'F277W'
    """
    if psf_type == "simple":
        gpsf = galsim.Gaussian(flux=1., sigma=sigma_psf * scale)
    elif psf_type == "mixture":
        raise NotImplementedError
    elif config.psf_type == "webbpsf":
        hdul = fits.open(psfstore)
        det_samp = hdul[1].header["DETSAMP"]
        psfim = hdul[1].data.astype(np.float64)
        pim = galsim.Image(np.ascontiguousarray(psfim), scale=scale/det_samp)
        gpsf = galsim.InterpolatedImage(pim)

    return gpsf


def make_psfstore(config, nradii=9):
    sigma = np.atleast_1d(config.sigma_psf)
    ngauss, nloc = len(sigma), 1

    from forcepho.patches.storage import PSF_COLS
    pdt = np.dtype([(c, np.float32) for c in PSF_COLS]
                   + [("sersic_bin", np.int32)])
    pars = np.zeros([nloc, nradii, ngauss], dtype=pdt)
    pars["amp"] = 1.0
    pars["Cxx"] = (sigma**2)[None, None, :]
    pars["Cyy"] = (sigma**2)[None, None, :]
    pars["sersic_bin"] = np.arange(nradii)[None, :, None]

    with h5py.File(config.psfstore, "a") as h5:
        bg = h5.create_group(config.band.upper())
        bg.create_dataset("parameters", data=pars.reshape(pars.shape[0], -1))
        bg.attrs["n_psf_per_source"] = pars.shape[1] * pars.shape[2]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # stamp
    parser.add_argument("--scale", type=float, default=0.03)
    parser.add_argument("--band", type=str, default="CLEAR")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    # scene
    parser.add_argument("--rhalf", type=float, default=0.2)
    parser.add_argument("--sersic", type=float, default=2.0)
    parser.add_argument("--flux", type=float, default=1.0)
    parser.add_argument("--dist_frac", type=float, default=1.5)
    # PSF
    parser.add_argument("--psf_type", type=str, default="simple")
    parser.add_argument("--sigma_psf", type=float, default=3.0)
    parser.add_argument("--psfstore", type=str, default="./single_gausspsf.h5")
    # MOre
    parser.add_argument("--snr", type=float, nargs="*", default=[20, 50])
    parser.add_argument("--add_noise", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="./")
    config = parser.parse_args()
    config.band = config.band.upper()

    try:
        os.remove(config.psfstore)
    except:
        pass

    # Make the image
    stamp = make_stamp(config.band, nx=config.nx, ny=config.ny, scale=config.scale)

    # Set the scene in the image
    scene = make_scene(stamp, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic)

    # render the scene
    psf = get_galsim_psf(config.scale, psf_type="simple", sigma_psf=config.sigma_psf)
    im = galsim_model(scene, stamp, psf=psf)

    for snr in config.snr:

        # --- compute uncertainty ---
        npix = np.pi*(scene[0]["rhalf"] / config.scale)**2
        signal = scene[0][config.band] / 2
        noise = signal / config.snr
        noise_per_pix = noise / np.sqrt(npix)
        unc = np.ones_like(im)*noise_per_pix
        noise = np.random.normal(0, noise_per_pix, size=im.shape)
        if config.add_noise:
            im += noise

        # --- write the test image ---
        hdul, wcs = stamp.to_fits()
        hdr = hdul[0].header
        hdr["FILTER"] = config.band
        hdr["SNR"] = config.snr
        hdr["DFRAC"] = config.dist_frac
        hdr["NOISED"] = config.add_noise

        image = fits.PrimaryHDU((im).T, header=hdr)
        uncertainty = fits.ImageHDU(unc.T, header=hdr)
        noise_realization = fits.ImageHDU(noise.T, header=hdr)
        catalog = fits.BinTableHDU(scene)
        catalog.header["FILTERS"] = ",".join([config.band])

        out = os.path.join(f"{config.outdir}", f"single_snr{snr:.0f}.fits")
        print(f"Writing to {out}")
        os.makedirs(os.path.dirname(config.outname), exist_ok=True)
        hdul = fits.HDUList([image, uncertainty, noise_realization, catalog])
        hdul.writeto(out, overwrite=True)
        hdul.close()

    # -- write the psf ---
    if config.psf_type == "simple":
        try:
            make_psfstore(config, nradii=9)
            print(f"wrote PSF info to {config.psfstore}")
        except(ValueError):
            print("PSF approx already exists")
            pass
