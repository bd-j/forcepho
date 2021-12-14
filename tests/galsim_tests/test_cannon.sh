#!/bin/bash
#SBATCH --mail-type=None               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --job-name=test-force        # Job name
#SBATCH --partition=gpu_test         # queue for job submission
#SBATCH --ntasks=1                   # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/forcetest_%A_%a.log    # Standard output and error log
#SBATCH --error=logs/forcetest_%A_%a.log    # Standard output and error log
pwd -P; hostname; date

module purge
module load git/2.17.0-fasrc01
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/11.4.2-fasrc01     # HeLmod latest
#module load cuda/10.1.243-fasrc01   # HeLmod
module load Anaconda3/5.0.1-fasrc01 # HeLmod w/o hdf5, matplotlib
#module load Anaconda3/5.0.1-fasrc02 # HeLmod w/ hdf5, matplotlib

export PROJECT_DIR=$SCRATCH/eisenstein_lab/$USER/force-test
cd $PROJECT_DIR
source activate fpho

task=${SLURM_ARRAY_TASK_ID}

# all the output will be stored here
outbase=./output/test_sampling_v1
# could add optimization flags here
extra_flags=""

# -- make and fit the galsim image ---
python fit_test.py --config_file ./test_config.yml \
                   --test_grid ./galsim_grid.fits \
                   --parameters_index $task --N_tasks 100 \
                   --sampling_draws 2048 --warmup 512 \
                   --maxfluxfactor 2 \
                   --outbase $outbase $extra_flags

echo "sample results at $outbase"

date