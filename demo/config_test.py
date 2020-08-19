#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py - Example configuration script for forcepho runs.
"""

import os
import numpy as np
from argparse import Namespace
config = Namespace()

# -----------
# --- Overall ----
config.logging = True
basedir = os.path.expandvars("$SCRATCH/eisenstein_lab/bdjohnson/phodemo/")
basedir = "."

# -----------------------
# --- Filters being run ---
config.bandlist = ["Fclear"]

# -----------------------
# --- Data locations ---
config.store_name = "galsim_galaxy_grid"
config.splinedatafile = "stores/sersic_mog_model.smooth=0.0150.h5"

config.store_directory = os.path.join(basedir, "stores/")
config.frames_directory = os.path.join(basedir, "data/")
config.raw_catalog = os.path.join(basedir, "data", "galsim_galaxy_grid_cat.fits")

sd, sn = config.store_directory, config.store_name
config.pixelstorefile = "{}/pixels_{}.h5".format(sd, sn)
config.metastorefile = "{}/meta_{}.h5".format(sd, sn)
config.psfstorefile = "{}/psf_{}.h5".format(sd, sn)

config.scene_catalog = os.path.join(basedir, "output", "test_scene.fits")

# ------------------------
# --- Data Types/Sizes ---
config.pix_dtype = np.float32
config.meta_dtype = np.float32
config.super_pixel_size = 8      # number of pixels along one side of a superpixel
config.nside_full = np.array([840, 840])         # number of pixels along one side of a square input frame

# -----------------------
# --- Patch Generation ---
config.max_active_fraction = 0.1
config.maxactive_per_patch = 15
config.patch_maxradius = 8

# -----------------------
# --- HMC parameters ---
config.sampling_draws = 256
config.warmup = [256, 256]
config.full_cov = True

# ------------------------
# --- PSF information ----
# Used for building PSF store
config.mixture_directory = "/Users/bjohnson/Projects/jades_force/data/psf/mixtures"
config.psf_fwhm = [2.0]  # single gaussian with FWHM = 2 pixels
config.psf_amp = [1.0]
