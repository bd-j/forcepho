# Demo: Basic

In this demo we make and fit a single source in a single exposure.

## `single_make.py`

This script uses GalSim to make a (noisy) image of a single galaxy in a single
band. The PSF is modeled as a single, symmetric Gaussian. The noise is modeled
as draws from an iid Gaussian in each pixel. Adjustable parameters in this
script include the flux, half-light radiius and Sersic parameter of the galaxy.
The S/N ratio  -- expressed as the S/N of the source within the half-light
radius -- can be adjusted. The pixel scale and PSF width (in pixels) are also
adjustable.  The script also makes (or augments) a file called
`single_gausspsf.h5` that contains the PSF data in forcepho format.  The final
FITS file has the following data model:

* `EXT1` - the GalSim model flux image, including added noise.
* `EXT2` - the flux uncertainty in each pixel.
* `EXT3` - the noise realization that was added to the GalSim model image.
* `EXT4` - A table of source parameters

In addition the header contains information about the WCS and the filter.

## `single_fit.py`

This script fits the source using forcepho in sampling mode.  For the initial
guess catalog this uses the table of true source parameters in the last
extension of the demo data FITS file, and thus does not test for initial burn-in
or optimization issues.  The interface demonstrated here is the simple FITS file
`patch` with communication to and kernel execution in the CPU (as opposed to the
GPU).

## `single_plot.py`

This script plots the data, residual, and model for the last iteration in the
chain as well as a trace plot and a corner plot for the source parameters.