# Configuration File


Many options and behavior of |Codename| are controlled by a configuration file,
which is in yaml format.  Here we give an example configuration file with
descriptions of each parameter.

Note that any parameter can generally be overridden at run time with a command
line argument. Forcepho will try to automatially expand shell variables.

Switches are generally represented with 0 (False, off) and 1 (True, on)

## Filters

```yaml
bandlist:
 - F435W
 - F606W
 - F775W
 - F814W
 - F850LP
 - F105W
 - F125W
 - F140W
 - F160W
```

This is a list of the bands for which images exist.  Fluxes will be measured for
these bands, and pixel data will be grouped by bands.  Only images with a header
keyword `FILTER` with value equal to one of these bands will be used in the
fitting.


## Input data locations

```yaml
raw_catalog:
  $PROJECT_DIR/data/catalogs/hlf_udf_v0.2.fits
big_catalog:

store_directory:
  $PROJECT_DIR/data/stores
splinedatafile:
  $PROJECT_DIR/data/stores/sersic_mog_model.smooth=0.0150.h5
pixelstorefile:
  pixels_hlf2_udf_bright.h5
metastorefile:
  meta_hlf2_udf_bright.json
psfstorefile:
  psf_hlf_ng4.h5
```

These are the locations of the initialization catalog (`raw_catalog`) as well as
the data stores `store_directory`. Within the store directory, the pixel and
met-data storage files are named, as well as the Gaussian mixture files.


## Output locations

```yaml
outbase:
  ./output/test
scene_catalog:
  outscene.fits
write_residuals:  # whether to output residual images
  1
```

All the output files will be placed within a directory specified by `outbase`.
See `output.md` for the structure of this directory. The output catalog of
parameter values after optimization or at the end of sampling will be placed in
this directory with the name given by `scene_catalog`. It is usually good
practice to give this directory a distinct name for each run.  The value of
`write_residuals` controls whether residual images (from the last parameter
state) are output for each patch.

## Bounds & Priors

```yaml
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
```

These parameters are used to specify limits on the parameter values.

The `add_barriers` switch can be used to add very steep prior penalty near the
edges, which is useful for the optimization methods that can otherwise get stuck
at the edges of the allowed parameter values

The entries under `bounds_kwargs` indicate allowed ranges for the parameters
sqrt(q) and pa.  The position ranges are allowed to move by `n_pix * pixscale`
arcseconds in both RA and Dec.

## Patch Generation

```yaml
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
```

These parameters control the checking out of regions and scenes that define
patches. The most important one is `maxactive_per_patch`, the maximum number of
sources to fit simultaneously in a patch.  It is generally limited by GPU memory
size.

## Sampling parameters

```yaml
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
```

These parameters control the HMC sampling.

## Optimization parameters

```yaml
use_gradients:
  1
linear_optimize:
  0
gtol:
  0.00001
```

These parameters control the optimization.  The most important one is
`linear_optimize`, which determines whether a final round of linear least
squares is used to optimize the fluxes, conditional on the best fit shapes and
positions.  This can be useful to overcome the effect of the 'barriers'
mentioned in the Bounds section, and also yields estimates for the flux
uncertainties and their covariance.

## Pre-processing

```yaml
original_images:  # search path
  $PROJECT_DIR/data/images/v2.0/hlsp*fits
cutID:
  udf
frames_directory:  # full path (optional, for preprocessing)
  $PROJECT_DIR/data/images/cutouts
max_snr:
  0
do_fluxcal:  # whether to flux calibrate the images using ABMAG keyword
  1
bitmask: # integer corresponding to the bits of the mask image that constitue "bad" pixels.
  1
frame_search_pattern:
  udf-??-??_*sci.fits
detection_catalog: # full path to input catalog
  $PROJECT_DIR/data/catalogs/detection_table_v0.5.fits
```

## Data Types & Sizes

```yaml
pix_dtype:
  float32
meta_dtype:
  float32
super_pixel_size:  # number of pixels along one side of a superpixel
  8
nside_full:  # number of pixels along one side of a square input frame
 - 2048
 - 2048
```

These will generally not need to be changed.

## Background tweaks

```yaml
tweak_background:
  tweakbg

# in nJy/pix, to be subtracted from individual exposures
tweakbg:
  F105W: -0.0511
  F125W: -0.0429
  F140W: -0.0566
  F160W: -0.0463
```

The value of `tweak_background` specifies the dictionary to use for background level tweaks.
Leave it empty if you don't want to do any background tweaks.
