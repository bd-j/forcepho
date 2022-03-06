# Demo 1

In this demo we simultaneously fit two nearby sources in a single exposure.

## `make_demo_image.py`

This script uses GalSim to make a (noisy) image of two galaxies in a single
band. The PSF is modeled as a simple, symmetric Gaussian. The noise is modeled
as draws from an iid Gaussian in each pixel. Adjustable parameters in this
script include the fluxes, half-light radii and Sersic parameters of each
galaxy, as well as the separation between the centers of the two galaxies
expressed as a fraction of the half-light radius of the first source. The S/N
ratio  -- expressed as the S/N of the first source within the half-light radius
-- can be adjusted. The pixel scale and PSF width (in pixels) are also
adjustable.  The script also makes (or augments) a file called
`single_gauss_psf.h5` that contains the PSF data in forcepho format.  The final
FITS file has the following data model:

* `EXT1` - the GalSim model flux in each pixel, including added noise.
* `EXT2` - the flux uncertainty in each pixel.
* `EXT3` - the noise realizstion that was added to the GalSim model image.
* `EXT4` - A table of source parameters

In addition the header contains inormation about the WCS and the filter.

## `demo_pair.py`

This script fits the pair of sources using forcepho in sampling mode.  For
initial guess catalog this uses the table of true source parameters in the last
extension of the demo data FITS file, and thus does not test for initial burn-in
or optimization issues.  The interface demonstrated here is the simple FITS file
`patch` with communication to and kernel execution in the CPU (as opposed to the
GPU).

