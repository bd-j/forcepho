#!/bin/bash
pwd -P; hostname; date

export PROJECT_DIR=$PWD
cd $PROJECT_DIR
source activate force

bands=( F090W F115W F150W F182M F200W F210M F277W F335M F356W F410M F444W F430M F460M F480M)
scales=( 0.03  0.03  0.03  0.03  0.03  0.03  0.06  0.06  0.06  0.06  0.06  0.06  0.06  0.06)

psfname=psf_jwst_2022-11-25_ng5m1
outdir=./output/cpu/${psfname}_noiseless
mkdir -p $outdir

for (( c=6; c<=13; c++ )) do
    SLURM_ARRAY_TASK_ID=$c
    band=${bands[$SLURM_ARRAY_TASK_ID]}
    scale=${scales[$SLURM_ARRAY_TASK_ID]}

    # -- make and fit the galsim image ---
    python psf_grid_fit.py  --parameter_grid ./psf_grid.yml \
                            --psfstore ./psf_data/jwst_2022-11-25/${psfname}.h5 \
                            --psfdir ./psf_data/jwst_2022-11-25/ \
                            --bandname $band \
                            --scales $scale \
                            --add_noise 0 \
                            --output_dir $outdir
done

date
