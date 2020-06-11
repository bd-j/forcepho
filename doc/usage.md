# fpho usage
How to use `fpho` with your images

## Data
This is the data that is required to be input

### Frames

Requirements for images are:

* They must all be the same dimension (for now), preferably 2048 by 2048.  Ideally they will be square but this is not a strict requirement.
* They must have dimensions that are an integer multiple of 8.
* The headers must contain a valid WCS (i.e. `"CRPIX"`, `"CRVAL"`, and `"CD"` or `"PC"` + `"CDELT"` matrix keywords.)
   (e.g. `astropy.wcs.WCS(astropy.io.fits.getheader(imname))` must not raise an error.)
* The headers of all frames must contain a `"FILTER"` keyword with string value, and preferably an `"ABMAG"` zeropoint keyword.
* There _must_ be a -pixel-matched uncertainty image associated with every science image, in same units as the science image.  Ideally the uncertainty image will have a filename that is easily generated from the science image filename (i.e. by a simple substitution)

Optionally each image may also have an associated background image and pixel mask image.

* The background image should be the same units as the science image, and will be subtracted from it during preprocessing.  If not supplied then it is assumed that the images are already background subtracted.
* The mask image can be an array of bitflags or a simple array of 0 (False, do not use this pixel) and 1 (True, use this pixel.)  If not supplied then the only pixels masked will be those with NaN or inf values for either the pixel data or the uncertainties.


Units

* Fluxes will be *reported* in image units (which should be 'per-pixel', not surface brightness)  Therefore, when fitting several images in a single FILTER it is highly desirable that all images be on the same flux scale.  If the `"ABMAG"` header keyword is present, pixel stores will automatically try to convert images to units of nJy/pixel.
* the WCS should map pixel indices to RA,DEC pairs, in decimal degrees


### Ancillary info

* The PSF must be known and provided.  It should be representable as a mixture of gaussians (there is code to compute this mixture from an image)
* There must be a catalog of initial (celestial or on-sky) positions.  This will be preprocessed/rectified to have a number of qualities.
* Ideally this catalog would contain rough estimates of the flux scale for the objects.
* You must obtain the splined Gaussian mixture amplitudes for Sersic profile approximations.  A default set of splines is provided, but is appropriate for a particular range of half-light radii.


## Preprocessing

To run `fpho`, several preprocessing steps are required.  Generally these will be accompished by a `preprocessing.py` script.  Since much of this is a script for converting your data into the types oif information required by `fpho`, it is user specific.

1. *Generate pixel data stores*
   
   This step produces HDF files of pixel data organized into superpixels, with backgrounds subtracted and masks applied.  Each `"FILTER"` will be contained in a separate HDF5 group, each frame will be a subgroup of that, with datasets corresponding to pixel data

2. *Generate metadata store*
   
   This step produces a json file containing all the header information, organized similarly to the pixel data but as a dictionary of dictionaries (keyed by `"FILTER"` and then by image name)

3. *Generate PSF store*
   
   This step creates an HDF5 file containing PSF information.  This information consists of a set of Gaussians describing the PSFs, as HDF datasets keyed


## Catalog

A catalog module will be used to convert your catalog to the format required by `fpho`.  This involves filling in any missing parameter initializations with initial guesses, converting positions to celestial positions, and other minor junk.


## The `config` file

This is where paths and options are specified.