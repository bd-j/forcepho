# forcepho

Galaxy photometry by image forward modeling, including image gradients.  The joint posterior probability distribution for the model image parameters can then be sampled using (H)MC algorithms.

The goal is to simultaneously forward model all relevant imaging data for a collection of nearby sources.  This should lead both to higher accuracy, usage of all information in the images, and better understanding of uncertainties than traditional techniques.

Galaxies and PSFs are represented by mixtures of Gaussians.

## Installation & Requirements

CPU operation requires a C compiler, HDF5 libraries, and a Python (ideally Anaconda) installation. See `requirements.txt` for minimum python package requirements.

GPU operation requires Nvidia GPU with compute capability >= 7.0 (developed for V100), a CUDA compiler, and the pycuda python package

1. create and activate a conda environment
   ```sh
   git clone http://github.com/bd-j/forcepho
   cd forcepho
   conda env create -f environment.yml
   conda activate force
   python -m pip install .
   ```

## Demos

See the `demo/` directory for several basic demos using the CPU kernel.
