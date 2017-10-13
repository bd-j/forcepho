Forcepho demos
=====

Mock data
---
These demos run on _mock_ data, where the data is generated from the same model used to fit them.

* `demo_mock_simplegm.py` Single Gaussian source and PSF, single stamp, mock generated from model, noise optionally added.  Useful for checking gradients, basic functionality.

* `demo_mock_f090w.py` Single point source with multi-Gaussian PSF (approximating f090w), mock generated from model, noise optionally added.  Useful for checking multi-gaussian implementations, and for testing the effects of the multi-gaussian PSF on the posterior distribution.

Simulated data
----
These demos run on _simulated_ data, which were generated using a detailed NIRCAM image simulator, Guitarra.  Thus it includes dead pixels, cosmic rays, Poisson noise etc.

* `demo_sim_pointsource.py` Single point source with multi-Gaussian PSF, fit to single-exposure simulated F090W data.  Useful for checking model performance on "real" data with known inputs.

* `demo_sim_all_pointsource.py`  Similar to above, but loop over all point-sources in the F090W simulated image.  Each point-source gets its own stamp, and so they are still fit in a single-source, single-stamp mode.  This is useful for investigating systematics, overall performance.

* `demo_sim_ps_twostamps.py`  Fit a single point-source to multiple simulated exposures in the same band.
