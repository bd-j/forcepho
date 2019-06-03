import sys, os, time
from copy import deepcopy
import numpy as np

from forcepho.data import PostageStamp
from forcepho.psf import PointSpreadFunction
from forcepho.sources import Galaxy, Scene
from forcepho.likelihood import WorkPlan, lnlike_multi
from forcepho.gaussmodel import rotation_matrix


# INFO USED TO CREATE THE TEST DATASET
NFILTER = 6
NEXPOSURE = 4
PATCH_SIZE = 9.0, 9.0 # in arcsec
EXPOSURE_SIZE = 9., 9.0 #in arcsec
NGALAXIES = 4

BANDS = "ABCDFGHJKLMPSTUV"[:NFILTER]
splinedata = 'test_data/sersic_mog_model.smooth=0.0150.h5'
TEST_PATTERN = "test_data/testscene*sci*2019.0524_14.52.fits"

rapadding = EXPOSURE_SIZE[0] - PATCH_SIZE[0]
SCENE_DEC = EXPOSURE_SIZE[1]/2.
SCENE_RA_RANGE = (rapadding / 2., rapadding /2. + PATCH_SIZE[0])
SCENE_DEC_RANGE = (SCENE_DEC - PATCH_SIZE[1]/2., SCENE_DEC + PATCH_SIZE[1]/2.)


def test_lnlike_and_grads():
    
    # read the test data
    stamps = read_patch()
    # Wrap the stamps in WorkPlan objects
    plans = [WorkPlan(stamp) for stamp in stamps]
    
    # create a scene with same number of galaxies and rough coordinate range as
    # used to create the test data.  Don't worry, we will explicitly set the
    # scene parameters below.
    scene = make_scene(NGALAXIES, SCENE_RA_RANGE, SCENE_DEC, bands=BANDS)
    
    # Choose an on-sky parameter vector (on line for each galaxy)
    Theta = np.array([2., 1.8, 1.6, 1.4, 1.2, 1., 1.8, 4.5, 0.7, 0.0, 1.45, 0.1,
                      2., 1.8, 1.6, 1.4, 1.2, 1., 3.6, 4.5, 0.7, 0.5, 1.45, 0.1, 
                      2., 1.8, 1.6, 1.4, 1.2, 1., 5.4, 4.5, 0.7, 1.0, 1.45, 0.1 , 
                      2., 1.8, 1.6, 1.4, 1.2, 1., 7.2, 4.5, 0.7, 1.5, 1.45, 0.1 ])
    
    # Calculate the likelihood and gradients
    # The "multi" here referes to multiple stamps or plans.
    lnp, lnp_grad = lnlike_multi(Theta, scene, plans)
    
    reference_lnp_grad = np.array([-5.22008894,   15.34495473, -7.67800619,  -48.40141636,
                                   -18.8533775,   43.707806, 1653.71492845,  1087.80221141,
                                   -157.39517522, 27.28817022, -6.0836077 ,   810.81031831,
                                    32.26244109, -25.60040544, 10.87526572,    13.63415805,
                                   -12.61946568,  14.68002691, -1179.66594919,  -805.48159405,
                                   -180.48482475, -104.88909734, -31.67633956,   695.10201124,
                                    3.67789212,   -7.21094036,    28.79090943,   -37.22654846,
                                   10.68695302,    6.51472907,  2506.59715348, -2411.54594921,
                                   53.67052832,    11.12234633,    -7.76881934,   887.69971767,
                                   42.69393912,   -14.10087602,   -58.41964388,    -8.85495786,
                                   93.25932384,    53.69945133,   803.30756057,   500.45862385,
                                   -215.89196001,    -4.7575051 ,   -52.27099567,  1004.9679265 ])
    
    assert np.isclose(lnp, -611529.2792642685)
    assert np.allclose(lnp_grad, reference_lnp_grad)


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


def read_patch(test_pattern=TEST_PATTERN,
               ralims=SCENE_RA_RANGE, declims=SCENE_DEC_RANGE):

    """Read data from all FITS files matching TEST_PATTERN,
    load the data into stamps while restricting to pixels that are within the
    ralims and declims
    
    :returns stamps:
        A list of PostageStamp objects
    """
    import glob
    files = glob.glob(test_pattern)
    # This is a list of postage stamp objects, where the image data is restricted to the patch of interest
    stamps = [read_exposure(fn, ralims, declims, mask=True) 
              for fn in files]

    return stamps


def read_exposure(fname, patchralims, patchdeclims, mask=True):
    """Read a single 'sci' FITS file corresponding to a single exposure and its 
    associated uncertainty image, and put into a PostageStamp object with
    correct metadata.  Then, restrict the pixels to those appearing within the
    sky coordinate limits given by `patchralims` and `patchdeclims`.  A PSF is
    added using get_pasf, which is identical to the function used to generate
    the test data.

    :param mask:
        If `True`, actually remove pixels not within the patch.  Otherwise,
        simply store a boolean flag in the `PostageStamp.good_pixel` ndarray that
        identifies which pixels are good.
        
    :returns stamp:
        A PostageStamp object
    """
    from astropy.io import fits
    
    hdr = fits.getheader(fname)
    data = fits.getdata(fname)
    unc = fits.getdata(fname.replace("sci", "unc"))
    
    s = PostageStamp()
    s.filtername = hdr["FILTER"]
    s.nx, s.ny = hdr["NAXIS1"], hdr["NAXIS2"]
    pixscale = hdr["PIXSCALE"]
    PA = hdr["ROT"]
    npsf = hdr["NPSF"]

    #  --- WCS ---
    s.scale = 1.0/pixscale * np.eye(2)
    s.dpix_dsky = np.matmul(s.scale, rotation_matrix(np.deg2rad(PA)))
    s.crpix = np.array([hdr["CRPIX0"], hdr["CRPIX1"]])
    s.crval = np.array([hdr["CRVAL0"], hdr["CRVAL1"]])

    # --- PSF ---
    s.psf = get_psf(npsf)

    # -- PIXEL DATA ---
    # x,y
    # note inversion here
    s.ypix, s.xpix = np.meshgrid(np.arange(s.ny), np.arange(s.nx))
    
    # restrict to pixels in patch, and reshape all images to 1D
    sky = pixelcoords_to_skycoords(s)
    inpatch = ((sky[0] > patchralims[0]) & (sky[0] < patchralims[1]) &
                (sky[1] > patchdeclims[0]) & (sky[1] < patchdeclims[1]))
    assert inpatch.sum() > 0

    if not mask:
        s.good_pixel = np.copy(inpatch)
        inpatch = slice(None)
    else:
        s.nx = inpatch.sum()
        s.ny = 1

    s.xpix = s.xpix.reshape(-1)[inpatch]
    s.ypix = s.ypix.reshape(-1)[inpatch]
   
    # fluxes and uncertainties within patch
    s.pixel_values = data.reshape(-1)[inpatch]
    s.ierr = 1. / unc.reshape(-1)[inpatch]
    
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


def pixelcoords_to_skycoords(stamp):
    pix = np.array([stamp.xpix.reshape(-1), stamp.ypix.reshape(-1)])
    sky = np.dot(np.linalg.inv(stamp.dpix_dsky), (pix.T - stamp.crpix).T).T + stamp.crval
    return sky.T