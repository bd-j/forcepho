#!/bin/bash

env_name=force
pymod=python/2.7.14-fasrc01
codedir=/n/home02/bdjohnson/codes/
rundir=/n/regal/eisenstein_lab/bdjohnson/

# --- CREATE CONDA ENVIRONMENT and add h5py ---
# This gets you:
#   numpy, scipy, matplotlib, astropy, h5py

module purge
module load gcc/7.1.0-fasrc01 openmpi/2.1.0-fasrc02 hdf5/1.10.1-fasrc01
module load $pymod
# --prefix $rundir
conda create -n $env_name --clone $PYTHON_HOME

echo "env created"

source activate $env_name
pip install --no-binary=h5py h5py

# --- pip install non-conda packages

pip install emcee
pip install corner
pip install theano
pip install pymc3

# --- intall other python packages from source---
repos=("joshspeagle/dynesty" "bd-j/forcepho")
branches=("master" "master")
n=${#repos[@]}

cd $codedir
for i in {0..$n}; do
    repo=${repos[$i]}
    pkg=${repo##*/}
    branch=${branches[$i]}
    if [ -a $pkg ]; then
	cd $pkg
	git pull origin master
    else
	git clone git@github.com:${repo}.git
	cd $pkg
    fi
    git checkout $branch
    python setup.py install
    cd ..
done

cd $rundir
git clone git@github.com:bd-j/xdf

# --- forcepho data files ---
# scp forcepho/forcepho/paths.py
# change fdir in paths.py
# mkdir forcepho/forcepho/mixtures/gauss_gal_results
# scp sersic*h5 to forcepho/forcepho/mixtures/gauss_gal_results
# --- xdf data files ---
# scp 3DHST*dat to xdf/data/catalogs
# scp xdf_f160-f814_3020-3470.fits to xdf/data/catalogs
# mkdir xdf/data/images
# wget xdf images from hlsp_*txt to xdf/data/images
