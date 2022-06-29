# JWST PSFs

Taken from jwst docs at https://stsci.app.box.com/v/jwst-simulated-psf-library on 10/2021. Linked from https://www.stsci.edu/jwst/science-planning/proposal-planning-toolbox/psf-simulation-tool and dated 08/2019

Current PSFs correspond to the OVERSAMP images

## From accompanying readme:

This directory contains simulated point spread functions for JWST, generated
with WebbPSF and using JWST’s predicted optical performance as represented by
the "predicted" OPD maps. For more information on the OTE contribution to the
overall wavefront error see https://webbpsf.readthedocs.io/en/stable/jwst.html

Each `*opd*.fits` file contains 4 image extensions:

- OVERSAMP - PSF oversampled by 4x relative to detector sampling.
- DET_SAMP - PSF binned to detector sampling. This will not be Nyquist sampled
  in all cases.
- OVERDIST - Oversampled PSF, modified based on expected geometric distortion at
  center of that instrument’s FOV
- DET_DIST - Detector sampled PSF, similarly modified based on the expected
geometric distortion.

Which PSF is most appropriate to use depends on your particular usage scenario.
An oversampled PSF is suitable for assessing the fundamental imaging performance
of JWST in cases with well-sampled data observations (either well-sampled
directly at that wavelength, or dithered and drizzled together to reconstruct
better-than-detector-resolution data.)  A detector-sampled PSF is suitable for
simulations of raw or un-dithered data products.  In general the effect of
geometric distortion is quite low, and you will see only minimal differences
between the `*SAMP` and `*DIST` images.

In some cases PSFs are modified based on the presence of additional masks in the
optical system, such as pupil stops. These are indicated in file names.

Additional information on PSF properties, such as variation of wavefront error
and distortion terms across the field of view or the effect of different input
spectral shapes on broadband PSFs, can be computed using WebbPSF. These example
PSFS are calculated for the center of each science instrument’s focal plane (or,
for NIRCam, at the center of SCA A1 for short wave filters and center of A5 for
long wave filters).

Notes:

1. More information about the different parameters used in these simulations can
   be found inside the fits file headers.
2. For the source spectrum, these calculations use a G0V stellar spectral type
   from the Castelli & Kurucz model libraries.
3. Calculations at large radii (> 500 lambda/D ~ 30 arcsec for 2 microns) will
   show numerical artifacts from Fourier aliasing and the implicit repetition of
   the pupil entrance aperture in the discrete Fourier transform. If you need
   accurate PSF information at such large radii, please contact Marshall Perrin
   or Marcio Melendez (melendez@stsci.edu) for higher resolution pupil data.


For more information follow this link:
https://webbpsf.readthedocs.io/en/stable/index.html