# Installation

## Requirements

The python packages that Forcepho requires are listed in `requirements.txt`.
Additional packages that may be necessary (especially for CUDA and MPI) are
listed in `optional-requirements.txt`.

In addition, for GPU and multiprocessing capability the python packages will
require CUDA and MPI installations (known to work with CUDA 10.1 and open-MPI).
You will also need an HDF5 installation.  These are often available on computing
clusters as modules.

Currently Forcepho is tested to work only with V100 Nvidia GPUs

## Conda install

The easiest way to install is to create a conda environment, using the supplied
`environment.yml` file.

```bash
git clone git@github.com:bd-j/forcepho.git
cd forcepho
conda env create -f environment.yml
conda activate force
python -m pip install .
```

## Clusters

### cannon

slurm script directives, also load these modules before installing:

```bash
module purge
module load git/2.17.0-fasrc01
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/11.4.2-fasrc01     # HeLmod latest
module load Anaconda3/5.0.1-fasrc01 # HeLmod w/o hdf5, matplotlib
source activate force
```

to update
```bash
git pull
module purge
module load intel/19.0.5-fasrc01
module load cuda/11.4.2-fasrc01 hdf5/1.10.5-fasrc01
module load Anaconda3/5.0.1-fasrc01
source activate force
python -m pip install .
```


### lux

install is easiest with a Miniconda download

```bash
module load cuda11.2 hdf5/1.10.6 gcc openmpi git slurm
conda env create -f environment.yml
source activate force
python -m pip install .
```

Note that to install mpi4py you should use pip (not conda) and you may have to
hide a linker that comes with miniconda

```bash
mv $HOME/miniconda3/envs/force/compiler_compat/ld $HOME/miniconda3/envs/force/compiler_compat/ld_old
python -m pip install --no-cache-dir mpi4py
```

For running slurm scripts just have this in the jobfile:

```bash
module purge
module load cuda11.2 hdf5/1.10.6 openmpi
source activate force
```

To update forcepho:

```bash
git pull
module purge
module load cuda11.2 hdf5/1.10.6 openmpi slurm git
source activate force
python -m pip install .
```

## GPU details

### lux

The user can start an MPS server

```bash
# Start MPS Daemon on both GPUs on this node
export CUDA_VISIBLE_DEVICES=0,1 # Select both GPUS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s accessible to the given $UID
nvidia-cuda-mps-control -d # Start the daemon.

```

### cannon

From the odyssey docs: While on GPU node, you can run `nvidia-smi` to get information about the assigned GPU

Not sure if it's necessary or how to enable MPS server.

### ascent/summit

MPS and Profiling

On ascent one enables MPS with

```bash
-alloc_flags "gpumps"
```

output.%h.%p

use `::KernelName:<int>` where `<int>` is the index of the kernel invocation that you want to profile

```bash
# detailed profiling of the kernel
jsrun -n1 -g1 -a1  nvprof --analysis-metrics -o /gpfs/wolf/gen126/scratch/bdjohnson/large_prof_metrics%h.%p.nvvp python run_patch_gpu_test_simple.py

# FLOP count
jsrun -n1 -g1 -a1  nvprof --kernels ::EvaluateProposal:1 --metrics flop_count_sp python run_patch_gpu_test_simple.py
```