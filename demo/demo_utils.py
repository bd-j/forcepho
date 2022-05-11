# -*- coding: utf-8 -*-

import sys, os, argparse
import numpy as np
import h5py

import galsim
from astropy.io import fits


__all__ = ["get_parser",
           "make_stamp", "make_scene",
           "galsim_model", "get_galsim_psf",
           "make_psf_store", "compute_noise_level",
           "write_to_disk"]


def get_parser():

    parser = argparse.ArgumentParser()
    # stamp(s)
    parser.add_argument("--scales", type=float, nargs="*", default=[0.03])
    parser.add_argument("--bands", type=str, nargs="*", default=["CLEAR"])
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    # scene
    parser.add_argument("--rhalf", type=float, nargs="*", default=[0.2, 0.2], help="arcsec")
    parser.add_argument("--sersic", type=float, nargs="*", default=[2.0])
    parser.add_argument("--flux", type=float, nargs="*", default=[1.0])
    parser.add_argument("--dist_frac", type=float, default=1.5)
    # PSF
    parser.add_argument("--sigma_psf", type=float, nargs="*", default=[3.0], help="in pixels")
    parser.add_argument("--psfstore", type=str, default="./single_gauss_psf.h5")
    # MOre
    parser.add_argument("--snr", type=float, default=50, help="S/N within rhalf")
    parser.add_argument("--add_noise", type=int, default=1)

    return parser


def make_stamp(band, scale=0.03, nx=64, ny=64, dither=(0, 0)):
    """Make a simple stamp instance

    Parameters
    ----------
    band : string
        The name of the filter for this stamp

    scale : float, arcsec
        The size of the pixels for this stamp

    nx : int
        Image size in x direction (number of columns)

    ny : int
        Image size in y direction (number of rows)

    dither : 2-tuple of float
        The dither in pixels (dx, dy)

    Returns
    -------
    stamp : instance of forcepho.slow.stamp.PostageStamp
        The stamp instance
    """
    from forcepho.slow.stamp import PostageStamp
    stamp = PostageStamp(nx, ny)
    stamp.scale = np.eye(2) / scale
    stamp.crval = [53.0, -27.0]
    stamp.crpix = np.array([(stamp.nx - 1) / 2 + dither[0],
                            (stamp.ny - 1) / 2 + dither[1]])
    stamp.filtername = band
    return stamp


def make_scene(stamps, rhalf=0.15, sersic=1, dist_frac=1.0, flux=1.0, nsource=1, pa=np.pi/2):
    """Convert a configuration namespace to a structured ndarray in the forcepho
    catalog format.

    Parameters
    ----------
    stamps : list of forcepho.slow.stamp.PostageStamp instance
        Defines the WCS.  Note that the output RA and Dec assume the source to
        be centered in the stamp.

    Returns
    -------
    cat : structured ndarray of shape (1,)
        The source parameters as a catalog row

    hdr : a FITS style header for the input stamp

    wcs : the WCS for the stamp
    """
    origin = 0
    nsource = max(len(np.atleast_1d(rhalf)), nsource)

    stamps = np.atleast_1d(stamps)
    bands = np.unique([stamp.filtername for stamp in stamps])

    # generate scene catalog
    from forcepho.superscene import sourcecat_dtype
    cdtype = sourcecat_dtype(bands=bands)
    cat = np.zeros(nsource, dtype=cdtype)

    # fill scene catalog
    cat["id"] = np.arange(nsource)
    cat["rhalf"] = np.array(rhalf)
    cat["sersic"] = np.array(sersic)
    cat["q"] = 0.9
    cat["pa"] = pa

    # add source fluxes (with color=0)
    for i, band in enumerate(bands):
        cat[band] = np.array(flux)

    # Add sources locations based on central coordinate of first stamp
    stamp = stamps[0]
    hdul, wcs = stamp.to_fits()
    ra, dec = wcs.all_pix2world((stamp.nx-1) / 2.,
                                (stamp.ny-1) / 2.,
                                origin, ra_dec_order=True)
    cat["ra"] = ra
    cat["dec"] = dec + np.arange(nsource) * (np.array(rhalf) * dist_frac) / 3600


    return cat


def galsim_model(scene, stamp, psf=None, verbose=False):
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

    hdul, wcs = stamp.to_fits()
    hdul.close()
    band = stamp.filtername
    pixel_scale = 1 / np.sqrt(np.linalg.det(stamp.scale))
    image = galsim.ImageF(stamp.nx, stamp.ny, scale=pixel_scale)

    for catrow in scene:
        gal = galsim.Sersic(half_light_radius=catrow["rhalf"],
                            n=catrow["sersic"], flux=catrow[band])
        # shift the galaxy
        x, y = wcs.all_world2pix(catrow["ra"], catrow["dec"], 0)
        dx, dy = x - (stamp.nx-1) / 2., y - (stamp.ny-1) / 2.
        if np.hypot(dx, dy) > 1e-2:
            offset = dx , dy
            if verbose:
                print(f"applying shift of {dx*pixel_scale}, {dy*pixel_scale}"
                      f" arcsec to {stamp.filtername}")
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


def make_psfstore(psfstore, band, sigma, nradii=9):
    """Make a PSF storage object for simple single Gaussian PSFs.
    """
    sigma = np.atleast_1d(sigma)
    ngauss, nloc = len(sigma), 1

    from forcepho.patches.storage import PSF_COLS
    pdt = np.dtype([(c, np.float32) for c in PSF_COLS]
                   + [("sersic_bin", np.int32)])
    pars = np.zeros([nloc, nradii, ngauss], dtype=pdt)
    pars["amp"] = 1.0
    pars["Cxx"] = (sigma**2)[None, None, :]
    pars["Cyy"] = (sigma**2)[None, None, :]
    pars["sersic_bin"] = np.arange(nradii)[None, :, None]

    with h5py.File(psfstore, "a") as h5:
        bg = h5.create_group(band.upper())
        bg.create_dataset("parameters", data=pars.reshape(pars.shape[0], -1))
        bg.attrs["n_psf_per_source"] = pars.shape[1] * pars.shape[2]


def compute_noise_level(scene, config):
    """Compute an image noise level corresponding to a given S/N ratio within
    the half-light radius (specified in config)
    """
    npix = np.pi*(scene[0]["rhalf"] / config.scales)**2
    signal = np.array([scene[0][b] for b in config.bands]) / 2
    noise = signal / config.snr
    noise_per_pix = noise / np.sqrt(npix)
    return noise_per_pix


def write_fits_to(out, im, unc, hdr, bands=[], noise=None, scene=None):
    """Write a FITS image with multuple extensions for image and uncertainty
    """
    image = fits.PrimaryHDU((im).T, header=hdr)
    uncertainty = fits.ImageHDU(unc.T, header=hdr)
    hdus = [image, uncertainty]
    if noise is not None:
        hdus.append(fits.ImageHDU(noise.T, header=hdr))

    if scene is not None:
        catalog = fits.BinTableHDU(scene)
        catalog.header["FILTERS"] = ",".join(bands)
        hdus.append(catalog)

    print(f"Writing to {out}")
    hdul = fits.HDUList(hdus)
    hdul.writeto(out, overwrite=True)
    hdul.close()


if __name__ == "__main__":

    parser = get_parser()
    config = parser.parse_args()


