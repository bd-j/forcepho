# Verification

Code and data in this directory is used to generate a forcepho image for a
particular set of source parameters and compare it to a reference image.  The
code used to generate the reference image is also included; this is based on the
reference python implementation and methods that exist in the `forcepho.slow`
module.

The `verify_config.yml` file contains information about file locations and how the
image is to be generated.

## Installation

Follow the forcepho installation instructions to install forcepho to a conda
environment named `force`.

## Make the reference image

Normally you won't need to do this; you'll simply compare a generated image *to*
the reference image and throw it away.  However, it might be useful to make more
reference images, or to test the parts of the code used for *generating* the
reference image.  The existing reference image was generated with:

```sh
python make_reference.py --sersic 2.2 --rhalf 0.1 --nx 63 --ny 32 --band F200W
```

Note that making a reference image requires an HDF5 file with PSF gaussian
mixture approximations.

## Generate new image and compare to the reference image

The code in `verify_reference.py` is used to generate a new image and compare it
(in terms of residuals) to the input image.  Currently the gradients or
likelihood values are not checked, but this will be improved in the future.

This verification will generally require a GPU, and the script can be run using
a slurm manager

```sh
sbatch verify_cannon.sh
```