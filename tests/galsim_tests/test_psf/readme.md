# Galsim Test: PSF

In this demo we make and fit single sources in single exposures but with
different parameters and at different S/N, and explore how the posterior
constraints change as a function of parameter and S/N, looking for biases.  We
true PSF images for the mock, and a Gaussian mixture PSF used in the model to
explore the effect of the PSF approximation.  The grid of parameters to test is
specified by a yaml file.  The code will use a GPU if one is available,
otherwise the CPU will be used.

```sh
# get some common info
ln -s ../test_plot.py test_plot.py
ln -s ../test_utils.py test_utils.py
ln -s ../../data/sersic_splinedata_large.h5 sersic_splinedata_large.h5
```

## PSF images

You will need PSF images and a PSF approximation.  The PSF images should be
named `<band>_psf.fits` and reside in a directory specified below by `--psfdir`.
If they are oversampled, the oversampling factor should be given in the keyword
`DETSAMP` in the header.

The PSF approximation should be in forcepho format, an HDF5 file with groups keyed by
`<band>`, each of which contains Gaussian mixture data.  It is specified by `--psfstore`

## Run the fit

```sh
# make and fit the test images
python test_psf_mixture.py  --test_grid ./test_psf_grid.yml \
                            --splinedatafile sersic_splinedata_large.h5 \
                            --add_noise 0 \
                            --psfstore psf_hst_ng4.h5 \
                            --psfdir ./mixtures/psf_images/hst/ \
                            --bandname F435W \
                            --scales 0.03 \
                            --dir ./output/cpu/hst_noiseless
```

This will make a directory `./output/cpu/hst_noiseless/` that is filled with
subdirectories named `<bandname>/`, each of which is filled with subdirectories
named  `<tag>` corresponding to each set of tested parameters, where `<tag>`
encodes the parameter values.  The directories will also contain a FITS table of
all the tested parameters, and several summary plots.

Obviously different Sersic mixture data can be supplied to the test, and a
different set of test parameters can be specified in a custom yaml file.

## Parameter subdirectories

Within each subdirectory `<band>` there will be

1. `<band>_psf.fits` - The FITS image used for the mock PSF.
2. `<psfstore>.h5` - The HDF5 file with Gaussian mixture approximations.

Within each subdirectory `<tag>` there will be

1. `<tag>_data.fits` - The mock data as a multi-extension FITS file.
2. `<tag>_samples.h5` - The output sampling chain and associated information
3. `<tag>_residuals.h5` - The output residuals from the last iteration of the
   chain, and associated information
4. `<tag_*.png>` - several diagnostic plots including MCMC trace, corner plot,
   and residuals.

## Data format

The mock data FITS file has the following data model:

* `EXT1` - the GalSim model flux image, including noise if added.
* `EXT2` - the flux uncertainty in each pixel.
* `EXT3` - (optional) the noise realization that was added to the GalSim model image.
* `EXT4` - A table of source parameters in forcepho format.

In addition the header contains required information about the WCS and the
filter (in the "FILTER" keyword).
