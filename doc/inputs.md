Inputs
======

|Codename| requires several ingredients to be on hand.  These are

* Imaging data
* Point Spread Functions (PSFs)
* An initial peak list
* A Gaussian mixture approximation to Sersic profiles.

Details about each aspect are given below.

Imaging data
------------

|Codename| expects imaging data with certain properties, which will be converted
to an internal storage format generally during "pre-processing".  These
properties are:

* The images must all be the same dimension (for now), preferably square.
  Default is 2048 by 2048.

* The images must have dimensions that are an integer multiple of 8.

* The headers must contain a valid WCS such as `"CRPIX"`, `"CRVAL"`, and `"CD"`
   or `"PC"` + `"CDELT"` matrix keywords.  In effect,
   `astropy.wcs.WCS(astropy.io.fits.getheader(imname))` must not raise an
   error.)  The WCS should map pixel indices (x, y) to (RA,DEC) pairs in decimal
   degrees.

* The headers of all images must contain a `"FILTER"` keyword with string value,
  though this can be added in pre-processing.

* Fluxes will be *reported* in image units (which should be 'per-pixel', not
  surface brightness)  Therefore, when fitting several different exposures in a
  single `FILTER` it is highly desirable that all images be on the same flux
  scale.  If the `"ABMAG"` header keyword is present for individual exposures,
  then pixel stores will automatically try to convert images to units of
  nJy/pixel.

* There _must_ be a pixel-matched uncertainty image associated with every
  science image.

Optionally each image may also have an associated background image and pixel
mask image:

* The background image should be the same units as the science image, and will
  be subtracted from it during preprocessing.  If not supplied then it is
  assumed that the images are already background subtracted, though global
  offsets may be supplied at runtime.

* The mask image can be an array of bitflags or a simple array of 0 (False, do
  not use this pixel) and 1 (True, use this pixel.)  If not supplied then the
  only pixels masked will be those with NaN or inf values for either the pixel
  data or the uncertainties.

The pre-processing script can be used to enforce the size constraints (e.g. by
padding or making cutouts), add the required header information, and parse
various file formats into the `ImageSet` structure, which are then used to
create the internal data storage.

PSFs
----

The PSFs used by |Codename| are approximations based on a Gaussian mixture model
fit to the actual PSFs.  Tools exist within forcepho to fit user-supplied PSFs
(appropriate for the supplied imaging data) with Gaussian mixture models.  The
PSFs, and the mixtures used to approximate them, can depend on detector
coordinates.

Initial peak catalog
--------------------

There must be a FITS binary table of initial (celestial or on-sky) positions.
This should have the following columns

* `ra` (decimal degrees)

* `dec` (decimal degrees)

* `q` (b/a, dimensionless) Use 0.8 if not known

* `pa` (radians, E of N) use 0.0 if not known

* `sersic` Sersic index (use 2 if not known)

* `rhalf` (arcsec) half-light radius estimate

* `<band>` rough flux estimate in `<band>`, where `<band>` corresponds to the `"FILTER"` keywords

Other columns may be present. The header of the catalog should contain the
keyword `"FILTERS"` which is a string containing a comma separated list of all
the band names.


Sersic Profiles
---------------

A standard lookup table of Gaussian parameters is provided.