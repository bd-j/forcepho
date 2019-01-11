#!bin/sh

env_name=force
pymod=python/2.7.14-fasrc01
codedir=/n/home02/bdjohnson/codes/
envdir=/n/regal/eisenstein_lab/bdjohnson/

# --- CREATE CONDA ENVIRONMENT and add h5py ---
# This gets you:
#   numpy, scipy, matplotlib, astropy, h5py

module purge
module load gcc/7.1.0-fasrc01 hdf5/1.10.1-fasrc01
module load $pymod
conda create --prefix $codedir -n $env_name --clone $PYTHON_HOME

source activate $env_name
pip install --no-binary=h5py h5py

# --- pip install non-conda packages

pip install emcee
pip install corner
pip install theano
pip install pymc3

# --- intall other python packages from source---
repos="joshspeagle/dynesty bd-j/forcepho bd-j/xdf"
branches="master master master"
n=${#repos[@]}

cd $codedir
for i in {1..$n}; do
    repo=${repos[$n]}
    pkg=${repo##*/}
    branch=${branches[$n]}
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
