import matplotlib.pyplot as pl
import numpy as np
import h5py

#from forcepho.mixtures.psf_mix_hmc import psf_model, psf_prediction
#from make_psf import read_image

from forcepho.slow.psf import PointSpreadFunction
from forcepho.slow.stamp import PostageStamp

from scipy.stats import multivariate_normal
from astropy.io import fits


def gaussian(XX, mean, covar):
    pass


pl.ion()

if __name__ == "__main__":
    pdatn = "../stores/psf_jwst_modsplit_jul22_mbs_ng4m0.h5"
    pdatn = "/Users/bjohnson/Projects/forcepho/tests/galsim_tests/test_psf/mixtures/psf_jwst_oct21_ng4m0.h5"
    ng = 4
    oversample = 8

    #with h5py.File(pdatn, "r") as pdat:
    pdat = h5py.File(pdatn, "r")
    #pdat.visit(print)

    bands = ["F150W", 'F200W', "F277W"]
    for band in bands: #= 'F277W'

        pscale = 0.03
        if pscale in ["F277W"]:
            pscale *= 2.0

        params = pdat[band]["parameters"][0, :ng]
        nx, ny = pdat[band].attrs["nx"], pdat[band].attrs["ny"]
        cx, cy = pdat[band].attrs["cx"], pdat[band].attrs["cy"]
        model = pdat[band]["model"][:].reshape(nx, ny)
        truth = pdat[band]["truth"][:]

        psf = PointSpreadFunction(parameters=params)

        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

        fig, ax = pl.subplots()
        ax.imshow(np.log(model.T), origin="lower")
        fig, ax = pl.subplots()
        rr = np.hypot(xx-cx, yy-cy).flatten()
        ax.plot(rr*0.03 / 4, truth.flatten(), ".")
        ax.set_yscale("log")

        data = np.array([rr * pscale / 4, truth.flatten()])
        fits.writeto(f"{band.lower()}_webbpsf.fits", data, overwrite=True)

        xx = (xx - cx) / oversample
        yy = (yy - cy) / oversample
        #sys.exit()
        im = np.zeros_like(xx)
        XX = np.array([xx, yy]).T
        for i in range(psf.n_gauss):
            mvn = multivariate_normal(mean=psf.means[i], cov=psf.covariances[i])
            im += psf.amplitudes[i] * mvn.pdf(XX)

        fig, ax = pl.subplots()
        ax.imshow(np.log(im.T), origin="lower")

        rr = np.hypot(xx, yy).flatten()
        f = im.flatten()
        fig, ax = pl.subplots()
        ax.plot(rr*pscale, f, ".")
        ax.set_yscale("log")

        data = np.array([rr * pscale, f])
        fits.writeto(f"{band.lower()}_fpho.fits", data, overwrite=True)

        opd_hdu = fits.open(f"/Users/bjohnson/Projects/jades-misc/psf/2022-10-03_opd/PSF_NIRCam_2022-10-03_opd_filter_{band.upper()}.fits")
        opd = opd_hdu[0].data
        hdr = opd_hdu[0].header

        nx, ny = opd.shape
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        rr = np.hypot(xx, yy).flatten()
        ff = opd.flatten()
        

    pdat.close()
