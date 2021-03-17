## Setup

on ody:

```bash
srun --pty -p gpu_test -t 0-01:00 --mem 1000 --gres=gpu:1 /bin/bash
srun --pty -p gpu -t 0-05:00 --mem 1000 --gres=gpu:1 /bin/bash

module purge
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/10.1.243-fasrc01
module load Anaconda3/5.0.1-fasrc01

GROUP=eisenstein_lab
MYSCRATCH=${SCRATCH}/${GROUP}/${USER}
source activate jadespho

cd ${MYSCRATCH}/phodemo
```

on lux:

```sh
srun -N 1 --partition=comp-astro --account=comp-astro --pty bash -I
module load cuda10.1 python/3.6.7 hdf5
module load numpy scipy h5py
module load numba pycuda
module load astropy littlemcmc

module load openmpi mpi4py
```

## Test Data

Test against a simple image created from `galsim`:

```sh
# Make test images (and catalog) using GalSim
mkdir ./data
python make_galsim_image.py
# preprocess to create data stores in the format expected by forcepho
mkdir ./stores
python preprocess_galsim.py
```

## Correctness

Now generate the model image and save it

```sh
mkdir ./output
python model_galsim_image.py --outbase=./output
```

Then, to generate images of the residuals for the S/N=100 objects:

```sh
cd display
python inspect_residuals.py
```

## Sampling

Now sample the posterior and regenerate the image from the best fits for each object:

```sh
# fit to the galsim noisy image
mkdir output/run1
python sample_galsim_image.py --patch_dir=./output/run1 --logging
# reconstruct from the posteriors
python reconstruct_galsim_image.py --patch_dir=./output/run1 --output_dir=./output
# Compare recovered parameters (here flux) to input
cd display
python explore.py --patch_dir=../output/run1 --colname=Fclear
```

## Features

It would be nice to show

 1. The covariances between adjacent sources as a function of the ratio distance/rhalf
 2. The difference in posteriors between fitting multiple (different resolution) bands separately versus simultaneously.
 3. Biases due to color gradients, or unmodeled substructure.
 4. The impact of covariances.