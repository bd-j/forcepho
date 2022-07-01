#!/bin/bash
pwd -P; hostname; date

export PROJECT_DIR=$PWD
cd $PROJECT_DIR
source activate force

bands=( F090W F115W F150W F200W F277W F335M F356W F410M F444W )
scales=( 0.03  0.03  0.03  0.03  0.06  0.06  0.06  0.06  0.06 )

outdir=./output/cpu/jwst_ng4m0_noiseless
#rm -rf $outdir

for (( c=0; c<=8; c++ )) do
    SLURM_ARRAY_TASK_ID=$c
    band=${bands[$SLURM_ARRAY_TASK_ID]}
    scale=${scales[$SLURM_ARRAY_TASK_ID]}

    # -- make and fit the galsim image ---
    python test_psf_mixture.py  --test_grid ./test_psf_grid.yml \
                                --splinedatafile ./sersic_splinedata_large.h5 \
                                --psfstore ./mixtures/psf_jwst_oct21_ng4m0.h5 \
                                --psfdir ./psf_images/jwst/ \
                                --bandname $band \
                                --scales $scale \
                                --add_noise 0 \
                                --dir $outdir
done

date
