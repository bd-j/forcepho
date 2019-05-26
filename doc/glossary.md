* **Band** (see **filter**)

* **Basic Assumed Input**   See:
    * **Exposure**
    * **Peak**
    * Point Spread Function (**`PointSpreadFunction`**)

* **Exposure**
  
  An exposure is a single 2D image of the sky through a single **Filter**, consisting of flux measurements in each pixel, along with associated pixel-matched uncertainty information.  It is the basic data product of the image reduction team and one of the basic inputs to *forcepho*.  For JWST/NIRCam, the exposures will by 2048 x 2048 pixels.  For short wavelength (SW) data, each pixel is ~0.032 mas on a side, while for long wavelength (lambda > 2.3 micron; LW) data, each image is ~0.064 mas on a side.  We assume the exposure data is photometrically and astrometrically calibrated, background subtracted, and comes with metadata describing the astrometry and the photometric zeropoint, though this may be in an inconvenient format.  Ancillary data (i.e. HST) is likely to have significantly different size/shape characteristics.

  Many separate exposures may cover a given location on the sky, taken with different central coordinates, at different orientations, with different pixel scales, or through different filters.  Because, partitioning the data leads to significant inefficiencies in sampling, all of this overlapping data must be considered in a given **likelihood call**.

* **Filter** (i.e., band, passband)
  
  The JADES program will obtain imaging through 9 separate filters, with names like ``F200W`` (wide 2 micron filter).  HST CANDELS provides data through an additional 6 photometric passbands.  In the deepest regions a number of additional HST passbands may be available.  The flux in each filter is (the most important) parameter of each `Source`.

  Data through a given filter may also be split by time; for example, ``F200W_A`` and ``F200W_B`` for data taken early and late in the observation.  Such a split will aid in the identificaion of temporal variability, including that due to spurious unmodeled features in the data.

* **`ImageGaussian`**
  A collection of on-image parameters describing a Gaussian that is one part of the representation of `Source`. 

* **`GaussianImageGalaxy`**
  
  A collection of `ImageGaussians` describing a single `Source` in a single Stamp, after convolution with the PSF (see **`PointSpreadFunction`**).

* **Likelihood Call**
  A function call with a vector argument corresponding to the parameters of all (non-boundary) sources within a patch.  It returns the likelihood of the patch data given the parameters of the `Source`s in that patch (the **``Scene``**), as well as the gradients of that likeihood with respect to the ``Scene`` parameters.

* **Passband**  (see **filter**)

* **Patch**
   
  A patch is a region on the sky, defined in celestial coordinates.  It may not necessarily be square, but is often assumed so.  Many exposures and filters may cover a given patch, and all pixels within the sky region are part of the patch. Inference is conducted simultaneously for the parameters of all **`Source`**s within a patch (except for fixed border objects).

* **Peak**
  
  An identified location of a "significant" intensity maximum in the scene.  There may be several peaks per galaxy, but *forcepho* assigns a unique **`Source`** to each peak.  A list of the celestial positions of identified peaks is a basic assumed input to *forcepho*.  This sets the total number of `Source`s to be considered in the model.

* **Pixel**
  
  A detector pixel in a single exposure, described by its x,y coordinates in the detector plane. Each pixel will have an associated flux and uncertainty value.

* **`PointSpreadFunction`**
 
  A collection of means and covariances that describes a Gaussian mixture approximation to the true Point Spread Function (PSF). 
  
  The true PSF is specific to a given filter and may possibly vary as function of position in an exposure.  It is assumed known based on calibration data, and is a basic input to *forcepho*.

* **`PostageStamp`**  (See **stamp**)

* **`Scene`**
  
  A list of sources (including in-active boundary sources) within a patch.

* **`Source`**
  
  A collection of on-sky parameters for each varied source.  There are also associated methods for converting to a set of circularized on-sky Gaussians.  Sources may also hold information about the (spatially local) metadata of each exposure in which they appear, i.e. the pixel scale and orientation in that exposure at the source location.

  While source positions will be a fitted parameter, it is assumed that they will not move by more than a pixel or two, so exposure metadata and the association of patches or relevant pixels with sources should remain constant.

  Within a give patch a source may be *active*, in which case its parameters are allowed to vary or *inactive* in which case its parameters are assumed fixed (and the gradient of the image with respect to these parameters is zero).  *Inactive* sources are meant to account for sources on the edge.

* **Stamp** or **`PostageStamp`**
  
  A collection of pixels which satisfies the following properties:
  - Taken through a single **filter**
  - sharing a common WCS (orientation on the sky and pixel scale)
  - sharing a single, approximately constant, assumed known point spread function.

* **Super Pixel**
  A set of N pixels in an exposure that share metadata and that can be completely described by their fluxes, uncertainties, and offset from a central reference pixel.  should be well matched to GPU data model, like say 32 or 128 pixels per superpixel.