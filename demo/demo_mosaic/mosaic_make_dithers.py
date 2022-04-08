# -*- coding: utf-8 -*-

import sys
from itertools import product
import os, argparse
import numpy as np
import h5py

import galsim
from astropy.io import fits

from demo_utils import get_parser
from demo_utils import make_stamp, make_scene
from demo_utils import get_galsim_psf, galsim_model, compute_noise_level
from demo_utils import make_psfstore, write_fits_to


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
    hdul, wcs = stamp.to_fits()
    pixel_scale = 1 / np.sqrt(np.linalg.det(stamp.scale))
    image = galsim.ImageF(stamp.nx, stamp.ny, scale=pixel_scale)

    for i, catrow in enumerate(scene):
        gal = galsim.Sersic(half_light_radius=catrow["rhalf"],
                            n=catrow["sersic"], flux=catrow[band])
        # shift the galaxy
        x, y = wcs.all_world2pix(catrow["ra"], catrow["dec"], 0)
        dx, dy = x - (stamp.nx-1) / 2., y - (stamp.ny-1) / 2.
        if np.hypot(dx, dy) > 1e-2:
            print(f"applying shift of {dx*pixel_scale}, {dy*pixel_scale} arcsec to source {i}")
            offset = dx , dy
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


def compute_noise_level(scene, config):
    npix = np.pi*(scene[0]["rhalf"] / config.scale)**2
    signal = np.array([scene[0][b] for b in np.atleast_1d(config.band)]) / 2
    noise = signal / config.snr
    noise_per_pix = noise / np.sqrt(npix)
    return noise_per_pix

def write_fits(im, unc, hdr, config, noise=None, scene=None):


    image = fits.PrimaryHDU((im).T, header=hdr)
    uncertainty = fits.ImageHDU(unc.T, header=hdr)
    hdus = [image, uncertainty]
    if noise is not None:
        hdus.append(fits.ImageHDU(noise.T, header=hdr))

    catalog = fits.BinTableHDU(scene)
    catalog.header["FILTERS"] = ",".join([config.band])

    os.makedirs(config.outdir, exist_ok=True)
    out = os.path.join(f"{config.outdir}", f"dither_{i:02.0f}.fits")
    print(f"Writing to {out}")
    hdul = fits.HDUList([image, uncertainty, noise_realization, catalog])
    hdul.writeto(out, overwrite=True)
    hdul.close()


if __name__ == "__main__":

    parser = get_parser()
    parser.set_defaults(nx=128, ny=128,
                        sigma_psf=[1.5],
                        snr=15,
                        rhalf=[0.2, 0.2])
    parser.add_argument("--all_dithers", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--basename", type=str, default="dither")
    config = parser.parse_args()
    print(config)

    try:
        os.remove(config.psfstore)
    except:
        pass

    # Set up the dither pattern
    # This is from NIRCAM
    dithers = [(0, 0),
               (0.0352, 0.1387),
               (0.0640, 0.3093),
               (0.1067, 0.0640),
               (0.1707, 0.1707),
               (0.2027, 0.3413),
               (0.2777, 0.0320),
               (0.3093, 0.1387),
               (0.3414, 0.3093)]
    dithers = np.array(dithers) / config.scale

    # Make the images
    stamps = [make_stamp(config.bands[0], nx=config.nx, ny=config.ny,
                         scale=config.scales[0], dither=(dx, dy))
              for dx, dy in dithers]

    # --- Set the scene in the image(s) ---
    scene = make_scene(stamps, dist_frac=config.dist_frac,
                       rhalf=config.rhalf, sersic=config.sersic)

    # --- Make and store the psf ---
    psf = get_galsim_psf(config.scales[0], sigma_psf=config.sigma_psf[0])
    make_psfstore(config.psfstore, config.bands[0], config.sigma_psf[0], nradii=9)

    # --- Render the scene in each stamp ---
    images = []
    for i, stamp in enumerate(stamps):
        images.append(galsim_model(scene, stamp, psf=psf))

    # --- compute uncertainty ---
    noise_per_pix = compute_noise_level(scene, config)

    # --- write the dither images ---
    os.makedirs(config.outdir, exist_ok=True)
    for i, stamp in enumerate(stamps):
        band = stamp.filtername
        hdul, wcs = stamp.to_fits()
        hdr = hdul[0].header
        hdr["FILTER"] = band.upper()
        hdr["SNR"] = config.snr
        hdr["DFRAC"] = config.dist_frac
        hdr["NOISED"] = config.add_noise
        hdr["PSFSIG"] = config.sigma_psf[0]

        im = images[i]
        unc = np.ones_like(im) * noise_per_pix
        noise = np.random.normal(0, noise_per_pix, size=im.shape)
        if config.add_noise:
            im += noise

        out = os.path.join(f"{config.outdir}", f"{config.basename}_{i:02.0f}.fits")
        print(f"Writing to {out}")
        write_fits_to(config.outname, im, unc, hdr, config, noise=noise, scene=scene)
