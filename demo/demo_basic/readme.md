# Test: Basic

Generate a galsim image with specific parameters and fit it with forcepho.

## setup

```sh
# get some common info
ln -s ../demo_utils.py demo_utils.py
ln -s ../data/sersic_splinedata_large.h5 sersic_splinedata_large.h5
```

## usage

Generate image, fit it, and plot corner, trace, and residuals
```sh
python basic.py --sersic 1.6 --rhalf 0.08 --snr 100 --add_noise 0
python basic_plot.py tests/sersic1.6_rhalf0.080_snr100_noise0/sersic1.6_rhalf0.080_snr100_noise0
```