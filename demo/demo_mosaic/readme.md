# Demo: Dithers vs. Mosaics

In this demo we fit two nearby sources in a series of dithered exposures, in the
drizzled mosaic made from those dithers, or in a direct image with comparable
S/N and pixel scale to the mosaic.  Note that you must install `drizzlepac` and
`stsci` python packages to make the mosaic
(https://github.com/spacetelescope/drizzlepac/blob/master/README.md); this is
most easily done with pip.

Note that the PSF of the mosaic is different than that of the dithers or the
'deep' image due to the drizzling process, and this is not taken into account in
the fitting of the mosaic.

```sh
# get some common info
ln -s ../demo_utils.py demo_utils.py
ln -s ../data/sersic_splinedata_large.h5 sersic_splinedata_large.h5

# make dithers
python mosaic_make_dithers.py --snr 15 --add_noise 0
# drizzle dithers to mosaic
python mosaic_combine_dithers.py --pixfrac 0.8
# make mosaic-like deep single image
python mosaic_make_dithers.py --snr 45 --n_dither 1 --basename deep --add_noise 0

# fit the mosaic
python mosaic_fit.py --image_basename ./data/mosaic --outdir ./output/mosaic
# fit the dithers
python mosaic_fit.py --image_basename ./data/dither --outdir ./output/dither
# fit the deep image
python mosaic_fit.py --image_basename ./data/deep --outdir ./output/deep

# plot corner plot comparisons
python mosaic_plot.py
```

## `mosaic_make_dithers.py`

This script uses GalSim to make (noisy) images of two galaxies in a single band,
with sub-pixel offsets corresponding to dithers. The PSF is modeled as a single,
symmetric Gaussian. The noise is modeled as draws from an iid Gaussian in each
pixel. Adjustable parameters in this script include the fluxes, half-light radii
and Sersic parameters of each galaxy, as well as the separation between the
centers of the two galaxies expressed as a fraction of the half-light radius of
the first source. The S/N ratio  -- expressed as the S/N of the first source
within the half-light radius -- can be adjusted. The pixel scale and PSF width
(in pixels) are also adjustable.  The script also makes (or augments) a file
called `single_gauss_psf.h5` that contains the PSF data in forcepho format.  The
final FITS files have the following data model:

* `EXT1` - the GalSim model flux image, including added noise.
* `EXT2` - the flux uncertainty in each pixel.
* `EXT3` - the noise realization that was added to the GalSim model image.
* `EXT4` - A table of source parameters

In addition the header contains information about the WCS, the filter, and the S/N.

The dither pattern is a 9-point pattern taken from NIRCAM documentation, and
appropriate fro reconstructing properly sampled images from subsampled images
using `drizzle`.

## `mosaic_combine_dithers.py`

This script uses `drizzlepac` from STScI to combine the dithers produced by
`mosaic_make_dithers.py` into a single mosaic.  Adjustable parameters include `pixfrac`

## `mosaic_fit.py`

This script fits the pair of sources using forcepho in sampling mode. For the
initial guess catalog this uses the table of true source parameters in the last
extension of the demo data FITS file, and thus does not test for initial burn-in
or optimization issues.  The interface demonstrated here is the simple FITS file
`patch` with communication to and kernel execution in the CPU (as opposed to the
GPU).

The fit can be made either to the drizzled mosaic or to the set of dithered
exposures. This is controlled by the `fit_mosaic` argument.  Fitting the
dithered exposures takes considerably longer.

## `mosaic_plot.py`

This script shows corner plot for the fluxes and other select parameters of the
two sources from fits to both kinds of data.
