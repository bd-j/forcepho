#!/bin/bash
#SBATCH --job-name=fpho-verify # Job name
#SBATCH --partition=comp-astro       # queue for job submission
#SBATCH --account=comp-astro         # queue for job submission
#SBATCH --ntasks=1                   # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --time=00:15:00               # Time limit hrs:min:sec
#SBATCH --output=logs/fpho-verify_%j.log   # Standard output and error log
pwd; hostname; date

export LC_ALL=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

module purge
module load cuda10.2 hdf5/1.10.6
source activate force

export PROJECT_DIR=$PWD
cd $PROJECT_DIR

band=F200W
reference=./data/reference-2021Nov30_f200w_sersic\=2.2_rhalf\=0.10.fits

# model the reference scene, and check the image residuals
python verify_reference.py --reference_image $reference --bandlist $band


