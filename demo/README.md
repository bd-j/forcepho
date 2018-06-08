# Forcepho demos

| Im Type | Source Type | Source #    | PSF      | Band # |  Coords | Solver   | Filename |
| ------- | ----------- | ----------- | -------- | ------ | ------- | -------- | -------- |
| mock    | Point       | single      | mixtures | single | pixel   | opt      | `mock_one_point_single.py` |
| mock    | Gauss       | single      | gauss    | single | pixel   | opt      | `mock_one_gauss_single_gpsf.py` |
| mock    | Gauss       | multi       | gauss    | multi  | pixel   | opt      | `mock_many_gauss_multi_gpsf.py` |
| mock    | Gauss       | multi       | mixtures | multi  | pixel   | opt, hmc | `mock_many_gauss_multi.py` |
| mock    | Sersic      | single      | mixtures | single | pixel   | nest,hmc | `mock_one_sersic_single.py` |
| sim_cw  | Point       | single loop | mixtures | single | pixel   | opt      | `simcw_one_point_single.py` |
| sim_cw  | Point       | single loop | mixtures | multi  | sky     | nest     | `simcw_one_point_multi.py` |
| sim     | Point       | single loop | mixtures | single | sky     | nest,hmc | `sim_one_point_single.py` |
| sim     | FixSersic   | multi       | mixtures | single | pixel   | hmc      | `mock_gauss_multi_psf.py` |


## Mock data

These demos run on _mock_ data, where the data is generated from the same model used to fit them.

* `demo_mock_ps_gmpsf.py` Single point source with multi-Gaussian PSF (approximating f090w), mock generated from model, noise optionally added.  Useful for checking multi-gaussian implementations, and for testing the effects of the multi-gaussian PSF on the posterior distribution.  Optimization only

* `demo_mock_simplegm.py` One single-Gaussian source and single-Gaussian PSF, one stamp, mock generated from model, noise optionally added.  Useful for checking gradients, basic functionality.  Optimization only

* `demo_mock_simplegm_multi.py` Same as above but for multiple single-Gaussian sources in multiple bands and/or stamps. Optimization or HMC.

* `demo_mock_simplegm_psf_multi.py` Same as above but with Gaussian mixture PSFs.  Reads a scene + data initialization from `catalog_for_demo.dat`

## Simulated data

These demos run on _simulated_ data, which were generated using a detailed NIRCAM image simulator, `Guitarra`, run through the `NCDHAS` reduction software.  Thus it includes dead pixels, cosmic rays, Poisson noise, non-linearity, etc.

* `demo_sim_all_pointsource.py` Single point source with multi-Gaussian PSF, fit to single-exposure simulated F090W data. Loop over all point-sources in the F090W simulated image.  Each point-source gets its own stamp, and so they are still fit in a single-source, single-stamp mode.  This is useful for  checking model performance on "real" data with known inputs, investigating systematics.  Optimization only.

* `demo_sim_ps_multiband_sampling.py`  Fit a single point-source to multiple (dithered) simulated exposures possibly in the multiple bands, looping over all sources in an image. Use nested sampling of the posterior instead of optimization.

* `demo_sim_ps_sandro_sampling.py` As above, but using the updated `guitarra` simulated images from S. Tachella which have correct noise estimates but include background and some different keywords.

* `demo_sim_extended_pixelspace.py` Fit a simulated single-band 3-source scene using GMs, with fixed `nsersic` and `r_h`.  Also, fir coordinates in the pixel space.
