PSF approximations
==================

For reasons of speed, Forcepho uses Gaussian mixture approximations to the PSF
instead of the actual PSF.  This requires generating these mixtures for a given
image with known (or empirically estimated) PSF.

In principle any procedure you want can be used to generate these
approximations, as long as the are provided in the format expected by forcepho
(see below).  Here we include a tool to generate these mixtures through an MCMC
exploration of GMM fit to the target image, using numpyro.

This tool requires an input PSF image.  This image can be oversampled (i.e. have
multiple pixels per final science image pixel) to aid in the characterization of
(mildly) undersampled high frequency structure.

Usage
-----

The script will conduct a fit for single PSF image and store the highest
likelihood parameter sample in a group in the HDF5 file indicated by
`--outname`. The group will be named by the `band`.

You must supply a band or filtername used to identify this PSF and the name of
the PSF image (assumed to be FITS with the image data in extension 0, but this
is adjustable.)  You should also indicate an oversampling factor if it is not 1,
and you can remove the outer N pixels in the x and y directions using the
`--trim` keyword.

The user should indicate the number of Gaussians to use in the approximation, as
well as a 'snr', used to dial up or down the uncertainty employed in the fit.
High values of 'snr' may result in long burn-in times with only minimal
improvement in the accuracy of the mixture.

Finally, and output HDF5 file should be specified.  If it does not exist it will
be created, otherwise the new mixture will be added as a new datagroup keyed by
`band`.  This will, by default, overwrite any existing datagroup of that name.
Diagnositic figures will be saved alongside the HDF5 file, and multi-extension
FITS images containing the target image, the model, and a catalog of parameters
can be generated if `--fitsout=1`.


Putting this all together, a call might look like:

```sh
python make_psf.py --band f090w --psf_image psf_f090w.fits --oversample 4 --trim 512 \
                   --ngauss 4 --snr 20 \
                   --outname ./mixtures/psf_nircam_ng4m0.h5 --overwrite 1 \
                   --savefig --fitsout 1
```

There are additional options (related to sampling or additional fit complexity),
see the script for details.

Mixture Format
--------------

Forcepho expects mixtures in a particular format.  These are HDF5 files with
data groups keyed by band or filter name.  Each data group will contain a
`parameters` data set which is a structured array usually of shape `(nloc, nradii,ngauss)`.
`nloc` is the number of locations across the image where a PSF is
specified (usually 1), `nradii` is the number of Sersic mixture components
(usually 9) and `ngauss` is the number of PSF mixture components.

Each element of the array has the fields
`"amp", "xcr", "ycr", "Cxx", "Cyy","Cxy"`
in single precision, and as integer `sersic_bin` that facilitates more general
PSF mixtures where the number of Gaussian components changes depending on the
radius of the Sersic component.

See `make_psf.convert_psf_data()` for details.