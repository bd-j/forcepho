Basic Usage
===========

How to use `fpho` with your images:

First, you will need collect the appropriate imaging data, PSFs, and identify a
an initial list or catalog of 'peaks' to be fit. A pre-processing script will
generally be required to convert the imaging data into a format useable by
forcepho (a pixel-data store and meta-data store). This pre-processing script
can convert from multi-extension FITS, multiple files indicating flax and
uncertainty, or other formats into an `ImageSet`.  See ./inputs.md for details

Second, you will need to generate a *configuration file* with information about
data input and output locations and details for the fitting process.  See
./configuration.md for details

Then, the following steps will lead to output that can be post-processed.
See ./output.md for detals on post-processing


Steps
-----
(to be consolidated)

1. Create PSF mixtures for mosaic and/or individual exposures

2. Pre-Process (`preprocess.py`)

   This creates the HDF5 storage files for pixel and meta-data.
   If slopes are present, make separate stores for mosaic and slope pixels.

3. Big sources

   * Pre-process bright mosaic pixels (including a S/N cap)

   * Find and catalog bright sources in the mosaic (`bright.py`)

   * Optimize/fit bright sources to the bright object pixel store using big
     sersics (x4 or x5 the normal rhalf grid)

   * [?] Identify the _very_ big and bright objects (based on rhalf_opt > rhalf_max_normal)

   * [?] Generate and subtract the v-big object model from the original mosaic
         and exposure stores. Or, have some way to subtract them in an initial
         setup of the patch (`JadesPatch.subtract_fixed`)

   * [?] Down-weight pixels in the centers of v-big and/or bright objects based
         on residual significance. This means inflating errors within isophotes
         such that |chi| ~ 1 or 2)

   * Replace initialization sources _within_ bright object ROIs (i.e. isophotes)
     with bright objects themselves (part of 4.)

4. Create initial catalog (`smallcat.py`)

   Based on the input detection catalog, replacing sources in the big/bright object
   ROIs with the those objects, and optionally restricting to a region on the
   sky.  Makes some small tweaks to the detection catalog values (e.g. reversing
   sign of PA). Use the result as `raw_catalog` in the config file.

5. Background subtraction & optimization loop

   * Optimize sources in the small catalog (`optimize.py`) based on mosaic.

   * Fit for a residual background (`background.py`) in the mosaic.
     If significant, put resulting tweak values in config file.

   * Re-optimize bright sources? (step 3)

   * Look for objects missing in the initial catalog?

   * Re-optimize sources in the small catalog (`optimize.py`) based on mosaic.
    
   * Replace initialization catalog with the optimization results.  This is done with
     ```sh
     postprocess.py --root output/<run_name> --catname postop_catalog.fits --mode postop
     ```

6. Sample posterior for source properties.

7. Post-process to create residual images (if available), show chains, etc...
