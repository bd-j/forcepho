# Galsim Tests

Code and data in this directory can be used to make GalSim images and fit them
with forcepho, and plot results and comparisons from the fits.

The `test_config.yml` file contains information about file locations and how the
fit is to be conducted.

## Make the parameter grid

The parameter grid that we will try to fit is specified by `galsim_grid.yml`.
We convert this into a FITS table of parameters as follows:

```python
from argparse import Namespace
from make_galsim_image import get_grid_params
config = Namespace(test_grid="./galsim_grid.yaml")
params = get_grid_params(config, 1)
```

This will make a file called `galsim_grid.fits` continaing the input truth
table.

## Make GalSim images for a set of parameters

The code in `make_galsim_image.py` is used to make GalSim images.  Each
generated file includes a flux image, an uncertainty image, and a binary table
of source parameters.  In practice we will only make images on the fly just
before we fit them, but the module includes an example invocation for generating
an image with arbitrary parameters specified at the command line.

Note that curently all generated images have the source located at the center.
Thus for odd image dimensions the source is located at the center of a pixel,
while for even dimensions it is located between pixels (at the edge of a pixel)

The PSF to use is specified by `psfstorefile` and `psf_type`.  This assumes the
existence of an HDF5 file containing original (oversampled) PSF images for the
specified band, as well as Gaussian approximations.

## Fit the GalSim images with forcepho

The code in `fit_galsim_image` (and in `child.py`) is used to fit a generated
GaSim image (with extensions as listed above).  The main code will loop over
entries in the parameter grid described above, generating GalSim images and then
fitting them.  All of the results are stored to a single directory.

It is necessary to set the following environment variable:

```sh
export PROJECT_DIR=$PWD
```

In practice this code will be called from slurm job submission script.  E.g.

```sh
slurm --array=0-900:100 test_cannon.sh
```

will submit 10 jobs, each of which will generate 100 images from the parameter
grid and fit them.

## Plot results

The `plot_test.py` module contains code for visualizing the results and
comparing the posterior samples from the fit to the input truth table in various
ways.