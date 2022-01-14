#!/bin/bash
#SBATCH --mail-type=None               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --job-name=fpho-verify         # Job name
#SBATCH --partition=gpu_test           # queue for job submission
#SBATCH --ntasks=1                     # Number of MPI ranks
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # How many tasks on each node
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:10:00               # Time limit hrs:min:sec
#SBATCH --output=logs/fpho-verify_%j.log    # Standard output and error log
#SBATCH --error=logs/fpho-verify_%j.log    # Standard output and error log
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


band=F200W
reference=./data/reference-2021Nov30_f200w_sersic\=2.2_rhalf\=0.10.fits

# model the reference scene, and check the image residuals
python verify_reference.py --reference_image $reference --bandlist $band


