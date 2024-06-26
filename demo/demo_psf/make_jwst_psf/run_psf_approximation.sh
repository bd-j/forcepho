#!/bin/bash

# Make the webbpsf

conda activate force

tag=2022-11-25
wver=1.2.1
psfdir=./webbpsf-v${wver}/${tag}_opd/
workdir=../psf_data/webbpsf-v${wver}/jwst_$tag
outdir=./mixtures
mkdir -p $workdir
mkdir -p $psfdir
mkdir -p $outdir

while getopts ":sl" option; do
  case $option in
    s) bands=( f090w f115w f150w f200w f162m f182m f210m );
       trim=576;
       pscale=0.03;;
    l) bands=( f277w f356w f444w f250m f300m f335m f410m f430m f460m f480m );
       trim=256;
       pscale=0.06;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
    esac
done

ngauss=5
ngauss_neg=1
snr=20
outname="${outdir}/psf_jwst_${tag}-wpsfv${wver}_ng${ngauss}m${ngauss_neg}.h5"

options="--snr ${snr} --ngauss ${ngauss} --ngauss_neg ${ngauss_neg}"
options=$options" --fitsout 1 --savefig --outname ${outname}"

echo $options

for b in ${bands[*]}; do
    echo "working on ${b}"
    python webbpsf_to_fpho.py --indir $psfdir --outdir $workdir --bands $b
    bopt="--band ${b} --psf_image_name ${workdir}/${b}_psf.fits --sci_pix_scale $pscale --trim $trim"
    python make_psf.py $bopt $options
done

python split_bands.py --filename $outname