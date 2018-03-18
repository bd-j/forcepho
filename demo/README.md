# Forcepho demos


## Mock data

These demos run on _mock_ data, where the data is generated from the same model used to fit them.

* `demo_mock_ps_gmpsf.py` Single point source with multi-Gaussian PSF (approximating f090w), mock generated from model, noise optionally added.  Useful for checking multi-gaussian implementations, and for testing the effects of the multi-gaussian PSF on the posterior distribution.  Optimization only

* `demo_mock_simplegm.py` Single Gaussian source and PSF, single stamp, mock generated from model, noise optionally added.  Useful for checking gradients, basic functionality.  Optimization only

* `demo_mock_simplegm_multi.py` Same as above but for multiple Gaussian sources in multiple bands. Optimization only.

* `demo_mock_simplegm_hmc.py` Same as above but with the option for HMC sampling


* `demo_mock_simplegalaxy_multi.py` Multiple single-Gaussian sources and a multi-gaussian PSF, multiple stamps in multiple bands, mock generated from model, noise optionally added.  Uses HMC to sample the posterior.

## Simulated data

These demos run on _simulated_ data, which were generated using a detailed NIRCAM image simulator, Guitarra, run through the NCDHAS reduction software.  Thus it includes dead pixels, cosmic rays, Poisson noise etc.


* `demo_sim_ps_multistamp_sampling.py`  Same as above, but nested sampling of the posterior instead of optimization.

* `demo_sim_ps_multi_sampling.py`  Fit a single point-source to multiple (dithered) simulated exposures possibly in the multiple bands. Use nested sampling of the posterior.
