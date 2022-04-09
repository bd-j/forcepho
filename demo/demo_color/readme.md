# Demo: Multi-source Multi-resolution

In this demo we simultaneously fit one or two (nearby) sources in a two exposures that
represent different bands with different PSFs and pixel sizes.

```sh
# get some common info
ln -s ../demo_utils.py demo_utils.py
ln -s ../data/sersic_mog_model.smooth\=0.0150.h5 sersic_mog_model.smooth\=0.0150.h5

# make mock images with a pair of galaxies in blue and red bands
python color_make.py --add_noise 0
# fit the sources in the blue image
python color_fit.py --image_names blue_pair.fits
# fit the sources in the red image
python color_fit.py --image_names red_pair.fits
# fit the sources in the blue + red image
python color_fit.py --image_names blue_pair red_pair.fits
```

## `color_make.py`

This script uses GalSim to make a (noisy) image of one or two galaxies in two
exposures. The PSF is modeled as a single symmetric Gaussian, but the width of
this Gaussian is different for each band. Furthermore the pixel size may be
different for each band. The noise is modeled as draws from an iid Gaussian in
each pixel. Adjustable parameters in this script include the fluxes, half-light
radii and Sersic parameters of each galaxy, as well as the separation between
the centers of the two galaxies expressed as a fraction of the half-light radius
of the first source. The S/N ratio  -- expressed as the S/N of the first source
within the half-light radius -- can be adjusted. For each band the pixel scales
and PSF widths (in pixels) are also adjustable.  The script also makes (or
augments) a file called `pair_gausspsf.h5` that contains the PSF data for both
bands in forcepho format.  The final FITS files have the following data model:

* `EXT1` - the GalSim model flux image, including added noise.
* `EXT2` - the flux uncertainty in each pixel.
* `EXT3` - the noise realization that was added to the GalSim model image.
* `EXT4` - A table of source parameters

In addition the header contains information about the WCS and the filter.

## `color_fit.py`

This script fits the sources in one or both exposures simultaneously using
forcepho in sampling mode.  For the initial guess catalog this uses the table of
true source parameters in the last extension of the demo data FITS file, and
thus does not test for initial burn-in or optimization issues.  The interface
demonstrated here is the simple FITS file `patch` with communication to and
kernel execution in the CPU (as opposed to the GPU).

## `color_plot_together.py`

This script plots the data, residual, and model for the last iteration in the
chain as well as a corner plot for the colors of the two sources inferred when
fitting sources in all bands simultaneously.

## `color_plot_separate.py`

This script plots the data, residual, and model for the last iteration in the
chain as well as a corner plot for the (aperture) colors of the sources that
results when inferring the flux in each band separately.