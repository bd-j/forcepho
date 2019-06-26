import sys, os, time
import numpy as np

from forcepho import paths
from forcepho.data import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.sources import Galaxy, Scene
from forcepho.gaussmodel import rotation_matrix

# INFO USED TO CREATE THE TEST DATASET
NFILTER = 6
NEXPOSURE = 4
PATCH_SIZE = 9.0, 9.0 # in arcsec
EXPOSURE_SIZE = 9., 9.0 #in arcsec
NGALAXIES = 4

BANDS = "ABCDFGHJKLMPSTUV"[:NFILTER]
splinedata = 'test_data/sersic_mog_model.smooth=0.0150.h5'

rapadding = EXPOSURE_SIZE[0] - PATCH_SIZE[0]
SCENE_DEC = EXPOSURE_SIZE[1]/2
SCENE_RA_RANGE = (rapadding / 2., rapadding /2. + PATCH_SIZE[0])
SCENE_DEC_RANGE = (SCENE_DEC - PATCH_SIZE[1]/2., SCENE_DEC + PATCH_SIZE[1]/2.)


# --- Utilities ---

def make_test_dataset():

    scene = make_scene(NGALAXIES, SCENE_RA_RANGE, SCENE_DEC)
    exposures = make_exposures(scene, NEXPOSURE)
    write_exposures(exposures)
    
    maxcounts = np.max([s.pixel_values.max() for s in exposures])
    vmin = None # 0
    vmax = None #maxcounts/1.5
    
    import matplotlib.pyplot as pl
    fig, axes = pl.subplots(NFILTER, NEXPOSURE)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    [ax.imshow(ex.pixel_values.T, origin="lower", vmin=vmin, vmax=vmax) for ax, ex in zip(axes.flat, exposures)]
    return scene


def make_scene(ngal, ralims, dec, bands=BANDS, splinedata=splinedata):
    sources = []
    # Arrange sources along a line in RA
    ra = np.linspace(ralims[0], ralims[1], ngal + 2)[1:-1]
    for i, x in enumerate(ra):
        s = Galaxy(splinedata=splinedata, filters=list(bands))
        s.ra = x
        s.dec = dec
        s.sersic = 1.45
        s.rh = 0.10
        s.q = 0.7
        s.pa = np.mod(0.5 * i, np.pi)
        s.flux = (2 - np.linspace(0, 1, len(bands))).tolist()
        sources.append(s)

    scene = Scene(sources)
    return scene


def make_exposures(scene, bands=BANDS,
                   NEXPOSURE=NEXPOSURE, EXPOSURE_SIZE=EXPOSURE_SIZE):
    exposures = []
    pixscale = np.ones(NFILTER) * 0.03
    pixscale[int(NFILTER/2):] = 0.06
    for i, b in enumerate(bands):
        for j in range(NEXPOSURE):
            # offset each exposure in a given band
            offset = (np.zeros(2) + j*3) * pixscale[i]
            center = np.array(EXPOSURE_SIZE) / 2 + offset
            # rotate some exposures in each band
            pa = 30. * np.mod(j, 2)
            size = np.array(EXPOSURE_SIZE) / pixscale[i]
            print(i, j, b, pixscale[i], offset, pa, size)
            ex = make_exposure(scene, size, b, crval=center,
                               pixscale=pixscale[i], PA=pa)
            exposures.append(ex)
    return exposures
                               

def make_exposure(scene, npix, band, snr_max=10,
                  pixscale=0.05, PA=0.0, crval=[0, 0],
                  npsf=2, **extras):
    
    s = PostageStamp()
    s.filtername = band
    s.nx, s.ny = (np.zeros(2) + np.array(npix)).astype(int)

    #  --- WCS ---
    s.scale = 1.0/pixscale * np.eye(2)
    s.dpix_dsky = np.matmul(s.scale, rotation_matrix(np.deg2rad(PA)))
    s.crpix = s.shape / 2.0
    s.crval = np.zeros(2) + np.array(crval)

    # --- PSF ---
    s.psf = get_psf(npsf)

    # -- PIXEL DATA ---
    # x,y
    # note inversion here
    s.ypix, s.xpix = np.meshgrid(np.arange(s.ny), np.arange(s.nx))
    # fluxes from scene
    s.pixel_values = np.empty(s.shape)
    im = np.sum([source.render(s, compute_deriv=False)[0] for source in scene.sources], axis=0)
    s.pixel_values = im.reshape(s.nx, s.ny)
    # 1 / sigma
    sigma = np.zeros_like(s.pixel_values) + s.pixel_values.max() / snr_max
    s.pixel_values += np.random.normal(0, sigma)
    s.ierr = 1. / sigma.reshape(-1)


    # Header info
    s.hdr = {"FILTER": (band, "name of filter"),
             "PIXSCALE": (pixscale, "arsec/pixel"),
             "ROT": (PA, "rotation in degrees, centered on CRPIX"),
             "CRPIX0": (s.crpix[0], "x pixel corresponding to CRVAL"),
             "CRPIX1": (s.crpix[1], "y pixel corresponding to CRVAL"),
             "CRVAL0": (s.crval[0], "lon of CRPIX"),
             "CRVAL1": (s.crval[1], "lat of CRPIX"),
             "NPSF": (npsf, "Number of gaussians in the PSFs")
             }


    return s


def get_psf(npsf):
    """Simple gaussian point spread function.
    """
    psf = PointSpreadFunction()
    psf.units = "pixels"
    psf.ngauss = npsf
    psf.means = np.zeros([npsf, 2])
    psf.covariances = (0.5 * (np.arange(npsf) + 1))[:, None, None] * np.eye(2)
    psf.amplitudes = np.ones(npsf) / npsf
    return psf


def write_exposures(exps, base="testscene"):
    now = time.strftime("%Y.%m%d_%H.%M")
    from astropy.io import fits
    for i, exp in enumerate(exps):
        hdu = fits.PrimaryHDU(exp.pixel_values)
        for k, v in list(exp.hdr.iteritems()):
            hdu.header[k] = v
        hdu.header["BUNIT"] = ("counts", "flux")
        hdu.writeto("test_data/{}_{}{}{}_{}.fits".format(base, exp.filtername, i, "sci", now))
        hdu = fits.PrimaryHDU(1 / exp.ierr.reshape(exp.nx, exp.ny))
        for k, v in list(exp.hdr.iteritems()):
            hdu.header[k] = v
        hdu.header["BUNIT"] = ("counts", "flux uncertainty")
        hdu.writeto("test_data/{}_{}{}{}_{}.fits".format(base, exp.filtername, i, "unc", now))
