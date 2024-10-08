.. _configuration:

Configuration File
==================

Many options and behavior of Forcepho are controlled by a configuration file,
which is in yaml format.  Here we give an example configuration file with
descriptions of each parameter.

Note that any parameter can generally be overridden at run time with a command
line argument. Also, forcepho will try to automatially expand shell variables.
See :py:meth:`forcepho.config.read_config` for details.

Switches are generally represented with 0 (False, off) and 1 (True, on)

Filters
-------

This is a list of the bands for which fluxes will be measured. Pixel data will
be grouped by bands.  Only images with a header keyword ``"FILTER`` with value
equal to one of these bands will be used in the fitting.  PSFs must be available
for each of these bands.

.. code-block:: yaml

    bandlist:
     - F090W
     - F115W
     - F200W
     - F277W
     - F335M
     - F444W


Input data locations
--------------------

First we have the locations of the initialization peak catalog (``raw_catalog``)
as well as the Gaussian mixture files.

.. code-block:: yaml

    raw_catalog:
        $PROJECT_DIR/data/catalogs/initial_peak_catalog.fits
    big_catalog:

    splinedatafile:
        $PROJECT_DIR/data/stores/sersic_mog_model.smooth=0.0150.h5
    psfstorefile:
        $PROJECT_DIR/data/psfs/psf_jwst_ng4.h5

For the simpler direct FITS file interface, use the following for the list of files to include:

.. code-block:: yaml

    fitsfiles:
     - band0_exp0.fits
     - band0_exp1.fits
     - band1_exp0.fits
     - band2_exp0.fits

They must be in order by band, but otherwise the filenames are arbitrary.


For the efficient StorePatch data interface, use something like the following

.. code-block:: yaml

    pixelstorefile:
      $PROJECT_DIR/data/stores/pixels_deepfield.h5
    metastorefile:
      $PROJECT_DIR/data/stores/meta_deepfield.json

Replace these filenames with the result of your image pre-processing, and make
sure those files are present (or soft-linked) at the stated locations.


Output locations
----------------

.. code-block:: yaml

    outbase:
      ./output/test
    scene_catalog:
      outscene.fits
    write_residuals:  # whether to output residual images, or just samples.
      1


All the output files will be placed within a directory specified by ``outbase``.
See ``output.md`` for the structure of this directory. The output catalog of
parameter values after optimization or at the end of sampling will be placed in
this directory with the name given by ``scene_catalog``. It is usually good
practice to give this directory a distinct name for each run.  The value of
``write_residuals`` controls whether residual images (from the last parameter
state) are output for each patch.

Bounds & Priors
---------------

.. code-block:: yaml

    # Add priors that are steep near the edges of the prior to aid optimization
    add_barriers:
      0

    bounds_kwargs:
    n_sig_flux: 5.0  # Nsigma/snr at flux = 1/nJy
    sqrtq_range: # range of sqrt(b/a)
        - 0.4
        - 1.0
    pa_range:  # range of pa, radians
        - -2.0
        - 2.0
    n_pix: # number of pixels for dRA, dDec
        2
    pixscale: # pixelscale for dRA, dDdec
        0.03


These parameters are used to specify limits on the parameter values.

The ``add_barriers`` switch can be used to add very steep prior penalty near the
edges, which is useful for the optimization methods that can otherwise get stuck
at the edges of the allowed parameter values

The entries under ``bounds_kwargs`` indicate allowed ranges for the parameters
sqrt(b/a) and pa.  The position ranges are allowed to move by ``n_pix * pixscale``
arcseconds in both RA and Dec.

Patch Generation
----------------

.. code-block:: yaml

    maxactive_per_patch:  # max number of active sources per patch
        15
    strict:  # whether to be strict about including all 'linked' sources
        1
    patch_maxradius:  # in arcsec
        15
    max_active_fraction:  # maximum fraction of all sources that can be checked out at once
        0.1
    ntry_checkout:
        1000
    buffer_size:
        5e7


These parameters control the checking out of regions and scenes that define
patches. The most important one is ``maxactive_per_patch``, the maximum number of
sources to fit simultaneously in a patch.  It is generally limited by GPU memory
size.

Sampling parameters
-------------------

.. code-block:: yaml

    target_niter:  # require this many samples for each source
        256
    sampling_draws: # generate this many samples for each patch
        256
    warmup:  # spend this many iterations tuning the proposal covariance matrix
        - 256
    full_cov:  # Whether to estimate the dense proposal covariance matrix, or just the diagonal.
        True
    max_treedepth: # do not take more than 2^max_treedepth steps in each trajectory
        9


These parameters control the HMC sampling.

Optimization parameters
-----------------------

.. code-block:: yaml

    use_gradients:
        1
    linear_optimize:
        0
    gtol:
        0.00001

These parameters control the optimization.  The most important one is
``linear_optimize``, which determines whether a final round of linear least
squares is used to optimize the fluxes, conditional on the best fit shapes and
positions.  This can be useful to overcome the effect of the 'barriers'
mentioned in the Bounds section, and also yields estimates of the flux
uncertainties and their covariance.

Pre-processing
--------------

.. code-block:: yaml

    original_images:  # search path
        $PROJECT_DIR/data/images/original/*fits
    cutID:
        deepfield
    frames_directory:  # full path (optional, for preprocessing)
        $PROJECT_DIR/data/images/cutouts
    max_snr:
        0
    do_fluxcal:  # whether to flux calibrate the images using ABMAG keyword
        1
    bitmask: # integer corresponding to the bits of the mask image that constitue "bad" pixels.
        1
    frame_search_pattern:
        deepfield-??-??_*sci.fits
    detection_catalog: # full path to input catalog
        $PROJECT_DIR/data/catalogs/detection_table_v0.5.fits


Pre-processing scripts can take many different forms, and are not strictly part
of a given inference run, but it can be useful to have the preprocessing
configuration stored with the other parameters.

Data Types & Sizes
------------------

.. code-block:: yaml

    pix_dtype:
        float32
    meta_dtype:
        float32
    super_pixel_size:  # number of pixels along one side of a superpixel
        8
    nside_full:  # number of pixels along one side of a square input frame
        - 2048
        - 2048


These will generally not need to be changed.

Background tweaks
-----------------

.. code-block:: yaml

    tweak_background:
        tweakbg

    # in nJy/pix, to be subtracted from individual exposures
    tweakbg:
        F105W: -0.0511
        F125W: -0.0429
        F140W: -0.0566
        F160W: -0.0463

The value of ``tweak_background`` specifies the name of the dictionary in the
configuration file to use for background level tweaks. Leave it empty if you
don't want to do any background tweaks.
