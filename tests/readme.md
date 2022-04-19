# Tests

* `galsim_tests/` - This directory contains scripts and data for fitting
  `forcepho` models to grids of GalSim models, in order to understand any biases
  induced by the Gaussian mixture approximations.

* `verification/` - This directory contains scripts to generate a forcepho model
  and compare it to a reference model.

* `slow_tests/` - Some old code used to test the slow python based implementations.

## CI Testing

* check that you can build a mock image into a postage stamp

* check that numerical gradients == analytic gradients

* some kind of coordinate checking?

* check that generating a scene with various numbers of bands produces the proper number of parameters
