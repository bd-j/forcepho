# Correctness

Test against a simple image created from `galsim`

```sh
mkdir stores
mkdir data
# Make a test image (and catalog) using galsim
python make_test_image.py
# preprocess to create data stores in the format expected by forcepho
python preprocess_test.py
# Now generate the model image and save it
python model_test_image.py
```