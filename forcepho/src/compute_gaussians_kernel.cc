/* compute_gaussians.cc

This is the core code to compute a Gaussian mixture likelihood and derivative
Top-level code view:

Create chi^2 and d(chi2/dparam) accumulators in shared memory and zero them.

For each band:

    For each exposure:

        Create on-image Gaussians from on-sky parameters, put in shared memory
            ImageGaussian[NGalaxy*ImageGaussiansPerGalaxy]

        For each pixel:

            Load Image Data
            Loop over all ImageGaussians:
                Evaluate Gaussians to create Residual image, save it

            Compute local_chi2 from residual image for this pixel
                Save it somewhere

            Loop over Active Galaxy:
                Loop over Gaussian in this Galaxy:
                    Compute local_dchi_dp and accumulate
                Reduce local_dchi_dp over warp and atomic_add to shared dchi_dp for galaxy

When done with all exposures, copy the accumulators to the output buffer.
*/

//===============================================================================

#include "header.hh"
#include "patch.cc"
#include "proposal.cu"
#include "common_kernel.cc"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void ConvolveOneSource(int band, int exposure, int g, int n_psf_per_source,
                       Patch * patch,  Source * sources, ImageGaussian * imageGauss) {

    // g: Source number
    // Unpack the source
    Source *galaxy = sources+g;

    // There are n_psf_per_source ~ ngauss_source * ngauss_psf PSF components per exposure
    // We need to find where they start for this exposure
    int psfgauss_start = patch->psfgauss_start[exposure];

    // photometric calibration for this exposure
    float G = patch->G[exposure];

    // Source dependent coordinate reference values
    int d_cw_start = 4 * (patch->n_sources * exposure + g);
    int cr_start = 2 * (patch->n_sources * exposure + g);
    float crpix[2], crval[2];
    crpix[0] = patch->crpix[cr_start]; crpix[1] = patch->crpix[cr_start+1];
    crval[0] = patch->crval[cr_start]; crval[1] = patch->crval[cr_start+1];
    float smean[2];
    smean[0] = galaxy->ra  - crval[0];
    smean[1] = galaxy->dec - crval[1];
    //------------------------

    // Do the setup of the transformations
    // Get the transformation matrix
    matrix22 D, R, S;
    D = matrix22(patch->D+d_cw_start);
    R.rot(galaxy->pa);
    S.scale(galaxy->q);

    //And its derivatives with respect to scene parameters
    matrix22 dS_dq, dR_dpa;
    dS_dq.scale_matrix_deriv(galaxy->q);
    dR_dpa.rotation_matrix_deriv(galaxy->pa);

    PixGaussian sersicgauss;    // This is where we'll queue up the source info
    // G is the conversion of flux units to image counts
    sersicgauss.G = G;
    // T is a unit circle, stretched by q, rotated by PA, and then distorted to the pixel scale
    sersicgauss.T = D * R * S;
    // And now we have the derivatives of T wrt q and PA.
    sersicgauss.dT_dq  = D * R * dS_dq;
    sersicgauss.dT_dpa = D * dR_dpa * S;
    // This is the dsky/dpix matrix
    sersicgauss.CW = matrix22(patch->CW+d_cw_start);
    Av(sersicgauss.CW, smean); //multiplies CW (2x2) by smean (2x1) and stores result in smean.
    // and here is the galaxy center in pixel coorinates
    sersicgauss.xcen = smean[0] + crpix[0];
    sersicgauss.ycen = smean[1] + crpix[1];

    // loop over source * psf gaussians
    for (int p = 0; p < n_psf_per_source; p++){

        PSFSourceGaussian *psfgauss = patch->psfgauss+psfgauss_start + p;
        int tid =  g * n_psf_per_source + p;   // Thread number

        // now this stuff is source and psf component dependent
        int s = psfgauss->sersic_radius_bin;
        sersicgauss.covar = patch->rad2[s];
        sersicgauss.amp   = galaxy->mixture_amplitudes[s];
        sersicgauss.da_dn = galaxy->damplitude_dnsersic[s];
        sersicgauss.da_dr = galaxy->damplitude_drh[s] ;
        sersicgauss.flux = galaxy->fluxes[band];         //pull the correct flux from the multiband array

        // Do the convolution, including gradients
        GetGaussianAndJacobian(sersicgauss, *psfgauss, imageGauss[tid]);
    }
}


// This function sets up the gaussians for a single source and a single PSF
// But it's very inefficient if not being done in parallel
//

void ConvolveOne(int band, int exposure, int tid, int n_psf_per_source,
                 Patch * patch, Source * sources,  ImageGaussian * imageGauss) {

    // where are we?
    int g = tid / n_psf_per_source;       // Source number
    int p = tid - g * n_psf_per_source;   // Gaussian number

    // There are n_psf_per_source ~ ngauss_source * ngauss_psf PSF components per exposure
    // We need to find where they start for this exposure
    int psfgauss_start = patch->psfgauss_start[exposure];

    // photometric calibration for this exposure
    float G = patch->G[exposure];

    // -----------------------
    // Source specific stuff
    // ----------------------
    // Unpack the source.
    Source *galaxy = sources+g;

    // Source dependent coordinate reference values
    int d_cw_start = 4 * (patch->n_sources * exposure + g);
    int cr_start = 2 * (patch->n_sources * exposure + g);
    float crpix[2], crval[2];
    crpix[0] = patch->crpix[cr_start]; crpix[1] = patch->crpix[cr_start+1];
    crval[0] = patch->crval[cr_start]; crval[1] = patch->crval[cr_start+1];
    float smean[2];
    smean[0] = galaxy->ra  - crval[0];
    smean[1] = galaxy->dec - crval[1];

    // Do the setup of the transformations
    // Get the transformation matrix
    matrix22 D, R, S;
    D = matrix22(patch->D+d_cw_start);
    R.rot(galaxy->pa);
    S.scale(galaxy->q);

    //And its derivatives with respect to scene parameters
    matrix22 dS_dq, dR_dpa;
    dS_dq.scale_matrix_deriv(galaxy->q);
    dR_dpa.rotation_matrix_deriv(galaxy->pa);

    // --------------------------
    PixGaussian sersicgauss;    // This is where we'll queue up the source info

    // G is the conversion of flux units to image counts
    sersicgauss.G = G;
    // T is a unit circle, stretched by q, rotated by PA, and then distorted to the pixel scale
    sersicgauss.T = D * R * S;
    // And now we have the derivatives of T wrt q and PA.
    sersicgauss.dT_dq  = D * R * dS_dq;
    sersicgauss.dT_dpa = D * dR_dpa * S;
    // This is the dsky/dpix matrix
    sersicgauss.CW = matrix22(patch->CW+d_cw_start);
    Av(sersicgauss.CW, smean); //multiplies CW (2x2) by smean (2x1) and stores result in smean.
    // and here is the galaxy center in pixel coorinates
    sersicgauss.xcen = smean[0] + crpix[0];
    sersicgauss.ycen = smean[1] + crpix[1];

    // --------------------------------------
    // specific to the source and psf components
    // --------------------------------------
    // unpack the PSF
    PSFSourceGaussian *psfgauss = patch->psfgauss+psfgauss_start + p;

    int s = psfgauss->sersic_radius_bin;
    sersicgauss.covar = patch->rad2[s];
    sersicgauss.amp   = galaxy->mixture_amplitudes[s];
    sersicgauss.da_dn = galaxy->damplitude_dnsersic[s];
    sersicgauss.da_dr = galaxy->damplitude_drh[s] ;
    sersicgauss.flux = galaxy->fluxes[band];         //pull the correct flux from the multiband array

    GetGaussianAndJacobian(sersicgauss, *psfgauss, imageGauss[tid]);
        // g * n_psf_per_source + p]);
}


void CreateImageGaussians(int band, int exposure, Patch * patch, Source * sources,
                 ImageGaussian * imageGauss) {

    int n_psf_per_source = patch->n_psf_per_source[band]; //constant per band.

    // Here we loop over every ImageGaussian, which is the nradii * npsf * nsource list
    // This is inefficient, but in principle allows for more code to be shared with cuda kernel.
    //int n_gal_gauss = patch->n_sources * n_psf_per_source;
    //for (int tid = 0; tid < n_gal_gauss; tid++){
    //    ConvolveOne(band, exposure, tid, n_psf_per_source, patch, sources, imageGauss);
    //}

    // BUT we probably want to loop over sources, and then loop over npsf within ConvolveOne
    // but that would break re-usability between cuda and cpu as currently coded in cuda
    int n_gal = patch->n_sources;
    for (int sid = 0; sid < n_gal; sid++){
        ConvolveOneSource(band, exposure, sid, n_psf_per_source, patch, sources, imageGauss);
    }

}


class CResponse{
    public:
      double chi2 = 0;
      py::array_t<float> pdchi2_dp = py::array_t<float>(NPARAMS * MAXSOURCES);
};


CResponse EvaluateProposal(int THISBAND, long _patch, long _proposal) {

    // Here's where we'll put the output
    CResponse response;
    // FIXME: can we move this to the Response object somehow?
    py::buffer_info rbuf = response.pdchi2_dp.request();
    float *pdchi2_dp = (float *) rbuf.ptr;
    //float pdchi2_dp[NPARAMS*MAXSOURCES]; // This holds the derivatives for multiple galaxies
    float dchi2_dp[NPARAMS];   // This holds the derivatives for one galaxy
    double pchi2 = 0; // this holds the chi2

    // Get the patch set up
    Patch *patch = (Patch *)_patch;

    // The Proposal is a vector of Sources[n_sources]
    Source *sources = (Source *)_proposal;

    // Loop over bands.  this should be done with separate calls to enable parallelization
    // int N_bands = ....
    // for (int b = 0; b < N_bands; b++) {
    //    int THISBAND = b;

    int n_sources = patch->n_sources;
    int n_gal_gauss = patch->n_psf_per_source[THISBAND];
    int band_N = patch->band_N[THISBAND];
    int band_start = patch->band_start[THISBAND];
    int n_gauss_total = n_sources*n_gal_gauss;
    ImageGaussian *imageGauss;
    int ret = posix_memalign((void**) &imageGauss, 64, sizeof(ImageGaussian)*n_gauss_total);

    // Loop over exposures in this band
    for (int e = 0; e < band_N; e++) {
        int exposure = band_start + e;
        int npix = patch->exposure_N[exposure];

        // Do the convolution
        // imageGauss = (ImageGaussian *) (shared + shared_counter);

        CreateImageGaussians(THISBAND, exposure, patch, sources, imageGauss);

        // Compute model for each pixel
        for (int p = 0; p < npix; p ++) {
            int pix = patch->exposure_start[exposure] + p;

            // Get the data and compute the model for this one pixel
            float xp = patch->xpix[pix];
            float yp = patch->ypix[pix];
            PixFloat data = patch->data[pix];
            PixFloat ierr = patch->ierr[pix];
            PixFloat residual = ComputeResidualImage(xp, yp, data, imageGauss, n_gauss_total);

            // Did the CPU ask that we output the residual image?
            if(patch->residual != NULL)
                patch->residual[pix] = residual;

            // Compute chi2 and accumulate it
            residual *= ierr;   // Form residual/sigma, which is chi
            data *= ierr;
            //double chi2 = ((double) residual * (double) residual);
            // below computes (r^2 - d^2) / sigma^2 in an attempt to decrease the size of the
            // residual and avoid loss of significance, but has same derivative as r^2/sigma^2
            double chi2 = ((double) residual - (double) data);
            chi2 *= ((double) residual + (double) data);
            pchi2 += chi2;
            residual *= ierr;   // We want res*ierr^2 for the derivatives

            // Now we loop over Sources and compute the derivatives for each
            for (int gal = 0; gal < n_sources; gal++) {
                for (int j=0; j<NPARAMS; j++) dchi2_dp[j]=0.0f;
                ComputeGaussianDerivative(xp, yp, residual,  //1.
                        imageGauss+gal*n_gal_gauss, dchi2_dp, n_gal_gauss);

                for (int j=0; j<NPARAMS; j++) pdchi2_dp[gal*NPARAMS + j] += dchi2_dp[j];
            }
        } //end loop over pixels
    } // end loop over exposures
    free(imageGauss);

    // Fill the response
    printf("%f", pchi2);
    response.chi2 = pchi2;

    // response.
    return response;

}



PYBIND11_MODULE(compute_gaussians_kernel, m) {
    m.def("EvaluateProposal", &EvaluateProposal);
    py::class_<CResponse>(m, "CResponse")
        .def_readwrite("chi2", &CResponse::chi2)
        .def_readwrite("dchi2_dp", &CResponse::pdchi2_dp);
}

