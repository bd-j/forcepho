# forcepho

Galaxy photometry by image forward modeling, including image gradients.  The joint posterior probability distribution for the model image parameters can then be sampled using (H)MC algorithms.

The goal is to simultaneously forward model all relevant imaging data for a collection of nearby sources.  This should lead both to higher accuracy, usage of all information in the images, and better understanding of uncertainties than traditional techniques.

Galaxies and PSFs are represented by mixtures of Gaussians.

## Installation & Requirements

Requires Nvidia GPU (developed for V100), CUDA compiler, MPI & HDF5 libraries, and a Python (ideally Anaconda) installation.

1. create and activate a conda environment
   ```sh
   git clone http://github.com/bd-j/forcepho
   cd forcepho
   conda env create -f environment.yml
   conda activate force
   python -m pip install .
   ```