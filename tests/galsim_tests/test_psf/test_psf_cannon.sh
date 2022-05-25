#!/bin/bash
#SBATCH --mail-type=None             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --job-name=test-force        # Job name
#SBATCH --partition=gpu              # queue for job submission
#SBATCH --ntasks=1                   # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=3:00:00                    # Time limit hrs:min:sec
#SBATCH --output=logs/ftestpsf_%A_%a.log  # Standard output and error log
#SBATCH --error=logs/ftestpsf_%A_%a.log   # Standard output and error log
pwd -P; hostname; date

module purge
module load git/2.17.0-fasrc01
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/11.4.2-fasrc01     # HeLmod latest
module load Anaconda3/5.0.1-fasrc01 # HeLmod w/o hdf5, matplotlib

export PROJECT_DIR=$PWD
cd $PROJECT_DIR
source activate force

bands=( F435W F606W F775W F814W F850LP F105W F125W F140W F160W )
band=${bands[$SLURM_ARRAY_TASK_ID]}

outdir=./output/hst_noiseless
rm -rf $outdir

# -- make and fit the galsim image ---
python test_psf_mixture.py  --test_grid ./test_psf_grid.yml \
                            --splinedatafile ./sersic_splinedata.h5 \
                            --psfstore ./psf_hlf_ng4.h5 \
                            --bandname $band \
                            --add_noise 0 \
                            --dir $outdir


date
