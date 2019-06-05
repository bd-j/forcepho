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


def make_individual_stamp(hdf5_file, ii_filter, jj_exp, counter, psfpath=None, fwhm=3.0, background=0.0):

    # get meta data about exposure
    dict_info = dict(zip(hdf5_file[ii_filter][jj_exp].attrs.keys(), hdf5_file[ii_filter][jj_exp].attrs.values()))

    # add image and uncertainty data to Stamp, flipping axis order
    stamp = PostageStamp()
    stamp.pixel_values = np.array(hdf5_file[ii_filter][jj_exp]['sci']).T - background
    stamp.ierr = np.array(hdf5_file[ii_filter][jj_exp]['rms']).T
    mask = np.array(hdf5_file[ii_filter][jj_exp]['mask']).T
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
    hdr = json.loads(hdf5_file[ii_filter][jj_exp]['header'][()])
    stamp.wcs = WCS(hdr)

    # add the PSF
    stamp.psf = pointspread.get_psf(os.path.join(psfpath, hdf5_file[ii_filter]['psf_name'][0]), fwhm)

    # add extra information
    stamp.photocounts = dict_info['phot']
    stamp.full_header = None
    stamp.filtername = ii_filter
    stamp.band = counter

    return(stamp)


def set_scene(sourcepars, fluxpars, filters, splinedata=None, free_sersic=False):

    # get all sources
    sources = []
    for ii_gal in range(len(sourcepars)):
        gal_id, x, y, q, pa, n, rh = sourcepars[ii_gal]
        s = Galaxy(filters=filters, splinedata=splinedata, free_sersic=free_sersic)
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
    W[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))
    D_mat = 1.0/3600.0*np.matmul(W, CW_mat)
    return(CW_mat, D_mat)


def patch_conversion(patch_name, splinedata, psfpath, n_psf=9):

    # read file
    hdf5_file = h5py.File(patch_name, 'r')

    # create scene
    mini_scene = set_scene(hdf5_file['mini_scene']['sourcepars'][:], hdf5_file['mini_scene']['sourceflux'][:], hdf5_file['mini_scene']['filters'][:].tolist(), splinedata=splinedata)

    # make list of stamps

    stamp_list = []
    band_counter = 0

    for ii_filter in hdf5_file.keys():
        for jj_exp in hdf5_file[ii_filter].keys():
            if 'exp' in jj_exp:
                stamp_list.append(make_individual_stamp(hdf5_file, ii_filter, jj_exp, band_counter, psfpath=psfpath, fwhm=3.0, background=0.0))
                band_counter += 0

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
        counter = 0

        # loop over all stamps (i.e. filters and exposures)
        for ii_filter in hdf5_file.keys():
            for jj_exp in hdf5_file[ii_filter].keys():
                if 'exp' in jj_exp:
                    wcs = stamp_list[counter].wcs
                    CW_mat, D_mat = get_transform_mats(source, wcs)
                    CW_list.append(CW_mat)
                    D_list.append(D_mat)
                    psf_list.append(n_psf * [stamp_list[counter].psf])
                    crpix_list.append(stamp_list[counter].crpix)
                    crval_list.append(stamp_list[counter].crval)
                    G_list.append(stamp_list[counter].photocounts)
                    counter += 1

        source.stamp_scales = D_list
        source.stamp_psfs = psf_list
        source.stamp_cds = CW_list
        source.stamp_crpixs = crpix_list
        source.stamp_crvals = crval_list
        source.stamp_zps = G_list

    return stamp_list, mini_scene


'''

# read in patch

base = "/Users/sandrotacchella/Desktop/patch_construction/"

psfpath = os.path.join(base, "psfs", "mixtures")

patch_name = os.path.join(base, "test_patch.h5")


# load spline data

splinedata = os.path.join(base, "data/sersic_mog_model.smooth=0.0150.h5")


# number of PSFs, place holder

num_psf = 9


# convert patch into list of stamps and mini scene

list_of_stamps, mini_scene = patch_conversion(patch_name, splinedata, psfpath, n_psf=num_psf)

'''

