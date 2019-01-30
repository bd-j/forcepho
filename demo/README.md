# Forcepho demos

| Im Type | Source Type | Sources     | PSF      | Bands  |  Coords | Solvers      | Filename |
| ------- | ----------- | ----------- | -------- | ------ | ------- | ------------ | -------- |
| mock    | Gauss       | single      | gauss    | single | pixel   | opt          | `mock_one_gauss_single_gpsf.py` |
| mock    | Gauss       | multi       | mixtures | multi  | pixel   | opt,hmc      | `mock_many_gauss_multi.py` |
| mock    | Sersic      | single      | mixtures | single | pixel   | nest,hmc,hem | `mock_one_sersic_single.py` |
| sim_cw  | Point       | single loop | mixtures | single | pixel   | opt          | `simcw_one_point_single.py` |
| sim_cw  | Point       | single loop | mixtures | multi  | sky     | nest         | `simcw_one_point_multi.py` |
| sim     | Point       | single loop | mixtures | single | sky     | nest,hmc,hem | `sim_one_point_single.py` |
| sim     | FixSersic   | multi       | mixtures | single | pixel   | nest,hmc,hem | `sim_many_fixsersic_single_pix.py` |


## Mock data

These demos run on _mock_ data, where the data is generated from the same model used to fit them.

* `mock_one_gauss_single_gpsf.py` One single-Gaussian source and single-Gaussian PSF, one stamp, mock generated from model, noise optionally added.  Useful for checking gradients, basic functionality. Multiple backends possible.

* `mock_many_gauss_multi.py` Same as above but with multiple single-Gaussian sources in multiple bands and/or stamps and Gaussian mixture PSFs.  Reads a scene + data initialization from `catalog_for_demo.dat`

* `mock_one_sersic_single.py` One multi-gaussian Sersic source and gaussian mixture PSF, in one band/exposure.  Can be fit using either nested sampling or HMC.


## Simulated data

These demos run on _simulated_ data, which were generated using a detailed NIRCAM image simulator, `Guitarra`, run through the `NCDHAS` reduction software.  Thus it includes dead pixels, cosmic rays, Poisson noise, non-linearity, etc.

* `simcw_one_point_single.py` Single point source with multi-Gaussian PSF, fit to single-exposure simulated F090W data. Loop over all point-sources in the F090W simulated image.  Each point-source gets its own stamp, and so they are still fit in a single-source, single-stamp mode.  This is useful for  checking model performance on "real" data with known inputs, investigating systematics.  Optimization only.

* `simcw_one_point_multi.py`  Fit a single point-source to multiple (dithered) simulated exposures possibly in the multiple bands, looping over all sources in an image. Use nested sampling of the posterior instead of optimization.

* `sim_one_point_single.py` As above, but using the updated `guitarra` simulated images from S. Tachella which have correct noise estimates but include background and some different keywords.

* `sim_many_fixsersic_single_pix.py` Fit a simulated single-band 3-source scene using GMs, with fixed `nsersic` and `r_h`.  Also, fit coordinates in the pixel space instead of celestial coordinates.
