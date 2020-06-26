# Setup
```bash
srun --pty -p gpu_test -t 0-01:00 --mem 1000 --gres=gpu:1 /bin/bash
srun --pty -p gpu -t 0-06:00 --mem 1000 --gres=gpu:1 /bin/bash

module purge
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/10.1.243-fasrc01
module load Anaconda3/5.0.1-fasrc01

GROUP=eisenstein_lab
MYSCRATCH=${SCRATCH}/${GROUP}/${USER}
source activate jadespho

cd ${MYSCRATCH}/phodemo
```

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
# sample the output
python sample_test_image.py
```

Then, generate images of the residuals for the S/N=100 objects
```
python inspect_residuals.py
```