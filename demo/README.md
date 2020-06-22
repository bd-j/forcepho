# Correctness

Test against a simple image created from `galsim`

```sh
# Make a test image (and catalog) using galsim
mkdir data
python make_test_image.py
# preprocess to create data stores in the format expected by forcepho
mkdir stores
python preprocess_test.py
# Now generate the model image and save it
mkdir output
python model_test_image.py
```

Then, in python

```python
from astropy.io import fits
truth = fits.getdata("data/galsim_galaxy_grid_truth.fits")
noisy = fits.getdata("data/galsim_galaxy_grid_noisy.fits")
model = fits.getdata("output/galsim_galaxy_grid_force.fits")
unc = fits.getheader("data/galsim_galaxy_grid_noisy.fits")["NOISE"]
chi = (noisy - model) / unc
```