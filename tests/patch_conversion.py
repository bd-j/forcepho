'''
convert patch to forcepho stamp object and mini scene
'''

import os
import numpy as np
import h5py
import json

from forcepho.sources import Galaxy, Scene
from forcepho.data import PostageStamp
from forcepho import psf as pointspread

from astropy.wcs import WCS


__all__ = ["make_individual_stamp", "set_scene", "get_transform_mats", "patch_conversion"]


def make_individual_stamp(hdf5_file, filter_name, exp_name, psfpath=None, fwhm=3.0, background=0.0):

    # get meta data about exposure
    dict_info = dict(zip(hdf5_file['images'][filter_name][exp_name].attrs.keys(), hdf5_file['images'][filter_name][exp_name].attrs.values()))

    # add image and uncertainty data to Stamp, flipping axis order
    stamp = PostageStamp()
    stamp.pixel_values = np.array(hdf5_file['images'][filter_name][exp_name]['sci']).T - background
    stamp.ierr = 1.0 / np.array(hdf5_file['images'][filter_name][exp_name]['rms']).T
    mask = np.array(hdf5_file['images'][filter_name][exp_name]['mask']).T
    stamp.nx, stamp.ny = stamp.pixel_values.shape
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # masking bad pixels: set error to inf
    bad = ~np.isfinite(stamp.ierr) | (mask == 1.0)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = np.inf
    stamp.ierr = stamp.ierr

    # add WCS info to Stamp
    stamp.crpix = dict_info['crpix']
    stamp.crval = dict_info['crval']
    stamp.dpix_dsky = dict_info['dpix_dsky']
    stamp.scale = dict_info['scale']
    stamp.CD = dict_info['CD']
    stamp.W = dict_info['W']
    hdr = json.loads(hdf5_file['images'][filter_name][exp_name]['header'][()])
    stamp.wcs = WCS(hdr)

    # add the PSF
    stamp.psf = pointspread.get_psf(os.path.join(psfpath, hdf5_file['images'][filter_name][exp_name]['psf_name'][0].decode("utf-8")), fwhm)

    # add extra information
    stamp.photocounts = dict_info['phot']
    stamp.full_header = None
    stamp.filtername = filter_name
    stamp.band = hdf5_file['images'][filter_name].attrs['band_idx']

    return(stamp)


def set_scene(sourcepars, fluxpars, filters, splinedata=None, free_sersic=True):

    # get all sources
    sources = []
    for ii_gal in range(len(sourcepars)):
        gal_id, x, y, q, pa, n, rh = sourcepars[ii_gal]
        s = Galaxy(filters=filters.tolist(), splinedata=splinedata, free_sersic=free_sersic)
        s.global_id = gal_id
        s.sersic = n
        s.rh = rh
        s.flux = fluxpars[ii_gal]
        s.ra = x
        s.dec = y
        s.q = q
        s.pa = np.deg2rad(pa)
        sources.append(s)

    # generate scene
    scene = Scene(sources)

    return(scene)


def get_transform_mats(source, wcs):
    '''
    get coordinate transformation matrices CW and D
    '''

    # get dsky for step dx, dy = 1, 1
    pos0_sky = np.array([source.ra, source.dec])
    pos0_pix = wcs.wcs_world2pix([pos0_sky], 1)[0]
    pos1_pix = pos0_pix + np.array([1.0, 0.0])
    pos2_pix = pos0_pix + np.array([0.0, 1.0])
    pos1_sky = wcs.wcs_pix2world([pos1_pix], 1)
    pos2_sky = wcs.wcs_pix2world([pos2_pix], 1)

    # compute dpix_dsky matrix
    [[dx_dra, dx_ddec]] = (pos1_pix-pos0_pix)/(pos1_sky-pos0_sky)
    [[dy_dra, dy_ddec]] = (pos2_pix-pos0_pix)/(pos2_sky-pos0_sky)
    CW_mat = np.array([[dx_dra, dx_ddec], [dy_dra, dy_ddec]])

    # compute D matrix
    W = np.eye(2)
    W[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))**-1
    D_mat = 1.0/3600.0*np.matmul(W, CW_mat)

    return(CW_mat, D_mat)


def zerocoords(stamps, scene, sky_zero=(53.0, -28.0)):
    zero = np.array(sky_zero)
    for source in scene.sources:
        source.ra -= zero[0]
        source.dec -= zero[1]
        new_crv = []
        for crv in source.stamp_crvals:
            new_crv.append(crv - zero)
        source.stamp_crvals = new_crv
    
    for stamp in stamps:
        stamp.crval -= zero
        
        


def patch_conversion(patch_name, splinedata, psfpath, nradii=9):

    # read file
    hdf5_file = h5py.File(patch_name, 'r')

    # get filter list
    filter_list = hdf5_file['images'].attrs['filters']

    # create scene
    mini_scene = set_scene(hdf5_file['mini_scene']['sourcepars'][:], hdf5_file['mini_scene']['sourceflux'][:], filter_list, splinedata=splinedata)

    # make list of stamps
    stamp_list = []
    stamp_filter_list = []
    for filter_name in hdf5_file['images'].attrs['filters']:
        for exp_name in hdf5_file['images'][filter_name].attrs['exposures']:
            stamp_list.append(make_individual_stamp(hdf5_file, filter_name, exp_name, psfpath=psfpath, fwhm=3.0, background=0.0))
            stamp_filter_list.append(filter_name)

    # loop over sources to add additional information
    for ii_s in range(len(mini_scene.sources)):

        source = mini_scene.sources[ii_s]

        # set lists to be filled
        D_list = []
        psf_list = []
        CW_list = []
        crpix_list = []
        crval_list = []
        G_list = []

        # loop over all stamps (i.e. filters and exposures) to add source specific information
        for s in stamp_list:
            wcs = s.wcs
            CW_mat, D_mat = get_transform_mats(source, wcs)
            CW_list.append(CW_mat)
            D_list.append(D_mat)
            psfs = nradii * [s.psf]
            psf_list.append(psfs)
            crpix_list.append(s.crpix)
            crval_list.append(s.crval)
            G_list.append(s.photocounts)

        source.stamp_scales = D_list
        source.stamp_psfs = psf_list
        source.stamp_cds = CW_list
        source.stamp_crpixs = crpix_list
        source.stamp_crvals = crval_list
        source.stamp_zps = G_list

    # loop over stamps, count gaussian components in psf for each band
    npsf_list = []

    for filter_name in filter_list:
        idx_s = (np.array(stamp_filter_list) == filter_name)
        s = np.array(stamp_list)[idx_s][0]
        psfs = nradii * [s.psf]
        npsf = np.sum([p.ngauss for p in psfs])
        npsf_list.append(npsf)

    mini_scene.npsf_per_source = np.array(npsf_list, dtype=np.int16)

    return stamp_list, mini_scene



'''
# testing

# define path to PSF and filename of patch

base = "/Users/sandrotacchella/Desktop/patch_construction/"
psfpath = os.path.join(base, "psfs", "mixtures")
patch_name = os.path.join(base, "test_patch.h5")


# filename of spline data
splinedata = os.path.join(base, "data/sersic_mog_model.smooth=0.0150.h5")


# number of PSFs, place holder
nradii = 9


# convert patch into list of stamps and mini scene
list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, nradii=nradii)

'''
