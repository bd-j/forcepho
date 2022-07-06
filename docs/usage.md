Basic Usage
===========

How to use `fpho` with your images.

Getting ready
-------------
First, you will need collect the appropriate imaging data, PSFs, and identify a
an initial list or catalog of 'peaks' to be fit. A pre-processing script may be
used to convert the imaging data into an efficient format useable by forcepho (a
pixel-data store and meta-data store). See ./inputs.rst for details

Second, you will need to generate a *configuration file* with information about
data input and output locations and details for the fitting process.  See
./configuration.rst for details

A Gaussian mixture approximation to each relevant PSF must be generated, using
tools provided with forcepho.  These are stored in an HDF5 file, in data groups
keyed by ``FILTER``.

Then, the following steps will lead to output that can be post-processed.
See ./output.md for details on post-processing


Basic Fitting
-------------

The basic procedure requires several ingredients to be instantiated using the
information above. Examples are given below for the simple FITS file interface
with CPU Kernel.

1. A `SuperScene` that holds global parameter state and parameter bounds, and
   can be used to check out sub-scenes.

    ```python
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "final_scene.fits"),
                               roi=cat["roi"],
                               bounds_kwargs=bounds_kwargs,
                               target_niter=config.sampling_draws)
    ```

2. A `Patcher` object that organizes image pixel data and image meta data.

    ```python
    class Patcher(FITSPatch, CPUPatchMixin):
          pass

    patcher = Patcher(fitsfiles=config.image_names,
                      psfstore=config.psfstore,
                      splinedata=config.splinedatafile,
                      sci_ext=1,
                      unc_ext=2,
                      return_residual=True)
    ```

3. Then a loop can start that checks out sub-scenes, finds the relevant pixel
data, and constructs an object that can compute posterior probabilities:

    ```python
    patchID = 0
    # Draw scenes until all sources have gotten target number of iterations of HMC
    while sceneDB.undone:
        # Draw the sub-scene and associated information
        # A seed of -1 will choose an available scene at random
        region, active, fixed = sceneDB.checkout_region(seed_index=-1)
        bounds, cov = sceneDB.bounds_and_covs(active["source_index"])

        # Collect the pixel data and meta-data
        patcher.build_patch(region, None, allbands=bands)
        # Transfer data to device, subtract fixed sources, set up parameter transforms
        model, q = patcher.prepare_model(active=active, fixed=fixed,
                                        bounds=bounds, shapes=sceneDB.shape_cols)
    ```

4. Within the loop we will either do optimization or HMC sampling, and then check
   the scene back in. Here we do sampling using littlemcmc:

    ```python
        # run HMC, with warmup
        out, step, stats = run_lmc(model, q.copy(),
                                  n_draws=config.sampling_draws,
                                  warmup=config.warmup,
                                  z_cov=cov, full=True,
                                  weight=max(10, active["n_iter"].min()),
                                  discard_tuned_samples=True,
                                  max_treedepth=config.max_treedepth,
                                  progressbar=config.progressbar)

        # Add additional information to the output
        final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                              step=step, stats=stats, patchID=patchID)
        # Write results to disk
        write_to_disk(out, config.outroot, model, config)
        # Check in the scene with new parameter values
        sceneDB.checkin_region(final, fixed, config.sampling_draws,
                              block_covs=covs, taskID=0)
        # write current global parameter state to disk as a failsafe
        sceneDB.writeout()
        # increment patch number
        patchID += 1
    ```


All Steps
----------

Stepping back a bit, one might want to do an intial round of optimization of the
entire catalog, and then use that as initialization for a sampling phase.  The
steps to do such a full run might look like the following

1. Create PSF mixtures for mosaic and/or individual exposures

2. (optional) Pre-Process (`preprocess.py`)

   This creates the HDF5 storage files for pixel and meta-data.
   If slopes are present, make separate stores for mosaic and slope pixels.

3. Make catalog of initial peaks in forcepho format

   The initial peaks to be fit must be supplied in a FITS binary table in the
   appropriate format. See ./inputs.rst for details of this format.

4. Background subtraction & optimization loop

   * Optimize sources in the catalog (`optimize.py`) using mosaic data with a
     S/N cap.

   * (optional) Fit for a residual background (`background.py`) in the mosaic. If
     significant, put resulting tweak values in config file.

   * (optional) Look for objects missing in the initial catalog.

   * (optional) Re-optimize sources in the catalog (`optimize.py`) based on mosaic.

   * Replace initialization catalog with the optimization results, including
     flux uncertainty estimates  This is done with
     ```sh
     postprocess.py --root output/<run_name> --catname postop_catalog.fits --mode postop
     ```

5. Sample posterior for source properties (`sampling.py`).

6. Post-process to create residual images (if available), show chains, etc...
