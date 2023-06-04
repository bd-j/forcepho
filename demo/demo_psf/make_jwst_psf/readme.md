# Making PSFs and approximations thereto

## 1) generate the psfs using webbpsf

This puts them in the `./webbpsf/yyy-mm-dd/` directory.  It also makes a
wavefront trending plot from webbpsf for that month.  It makes PSFs for many wide and medium bands in both NIRCam channels

```sh
python webbpsf_opd.py --year 2022 --month 11 --day 25
```

## 2) Approximate the psfs with Gaussians

This loops over channels and the bands in each channel, and for each band copies
a single extension of the webbpsf output to a common work directory with a
simple name . It then runs the `make_psf.py` script which uses numpyro to
generate a Gaussian mixture approximation for that band.

This script can be adapted easily to work on other PSF images; if the images are
oversampled relative to the images to be fit then that oversampling factor must
be given in the header as `DET_SAMP`. The plate scale of the images to be fit is
specified as an input option.

```sh
# which PSFs are we working with?
tag=2022-11-25
psfdir=./webbpsf/${tag}_opd
# Where do we put the output?
workdir=../psf_data/jwst_$tag
mkdir -p $workdir
mkdir -p $psfdir

# set parameters for the mixture
ngauss=5
ngauss_neg=1
snr=50
outname="./mixtures/psf_jwst_${tag}_ng${ngauss}m${ngauss_neg}.h5"

options="--snr ${snr} --ngauss ${ngauss}"
options=$options" --fitsout 1 --savefig --outname ${outname}"
echo $options

# loop over channels and bands
channels=( s l )
for channel in ${channels[*]}; do
  case $channel in
    s) bands=( f090w f115w f150w f200w f162m f182m f210m );
       trim=576;
       pscale=0.03;;
    l) bands=( f277w f335m f356w f410m f444w f430m f460m f480m f250m f300m );
       trim=256;
       pscale=0.06;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
    esac

    # copy (part of) the webbpsf output to a working directory with simple filename
    for b in ${bands[*]}; do
        python webbpsf_to_fpho.py --indir $psfdir --outdir $workdir --bands $b
    done
    # generate approximations in the working directory
    for b in ${bands[*]}; do
        bopt="--band ${b} --psf_image_name ${workdir}/${b}_psf.fits"
        bopt=${bopt}" --sci_pix_scale $pscale --trim $trim"
        python make_psf.py $bopt $options
    done
done
```

A bash script `run_psf_approximation.sh` that does all this for the sw and lw channels run separately is included.  After modifying the relevant variables call it with

```sh
./run_opd_approximation -s
./run_opd_approximation -l
```
