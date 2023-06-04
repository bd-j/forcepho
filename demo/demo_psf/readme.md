# Test: PSF

Generate a galsim image with specific parameters and using a 'real' PSF, and fit
it with forcepho using a PSF approximation.  See `make_jwst_psf/` for an example of
the PSF approximation process.

## setup

```sh
# get some common info
ln -s ../demo_utils.py demo_utils.py
ln -s ../data/sersic_splinedata_large.h5 sersic_splinedata_large.h5
```

## usage

Generate image, fit it, and plot corner, trace, and residuals

```sh
python psf_one_fit.py --sersic 1.6 --rhalf 0.06 --snr 100 --add_noise 0 \
                      --psfimage ./psf_data/jwst_2022-11-25 \
                      --psfstore ./psf_data/jwst_2022-11-25/psf_jwst_2022-11-25_ng5m1.h5 \
                      --bandname F200W
```

This can be run over a grid of galaxy parameters using the `psf_grid_fit.py`
script; the grid itself is specified in a YAML file (e.g. `psf_grid.yml`).
Scripts for running the fits over the grid are provided (`run_psf*sh`)
