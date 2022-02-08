/* compute_gaussians.cc

This is the core code to compute a Gaussian mixture likelihood and derivative
Top-level code view:

Create chi^2 and d(chi2/dparam) accumulators in shared memory and zero them.

For each exposure:

    Create on-image Gaussians from on-sky parameters, put in shared memory
        ImageGaussian[NGalaxy*ImageGaussiansPerGalaxy]

    For one pixel per thread (taking BlockSize steps):

        Load Image Data
        Loop over all ImageGaussians:
            Evaluate Gaussians to create Residual image, save it

        Compute local_chi2 from residual image for this pixel
        Reduce local_chi2 over warp; atomic_add result to shared mem

        Loop over Active Galaxy:
            Loop over Gaussian in this Galaxy:
                Compute local_dchi_dp and accumulate
            Reduce local_dchi_dp over warp and atomic_add to shared dchi_dp for galaxy

When done with all exposures, copy the accumulators to the output buffer.
*/
// Shared memory is arranged in 32 banks of 4 byte stagger

//===============================================================================

#include "header.hh"
#include "patch.cc"
#include "proposal.cu"
#include "common_kernel.cc"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


void Convolve(int band, Patch * patch, Source * sources,
              int d_cw_start, int cr_start) {

    int n_psf_per_source = patch->n_psf_per_source[band]; //constant per band.
    int n_gal_gauss = patch->n_sources * n_psf_per_source;

    // Here we loop over every ImageGaussian, which is the nradii * npsf * nsourtce list
    // BUT we probably want to loop over 
    for (int tid = 0; tid < n_gal_gauss; tid++){
        // Unpack the source and gaussian.
        int g = tid / n_psf_per_source;       // Source number
        int p = tid - g * n_psf_per_source;   // Gaussian number

        Source *galaxy = sources+g;
        PSFSourceGaussian *psfgauss = patch->psfgauss+psfgauss_start + p;
        PixGaussian    sersicgauss;    // This is where we'll queue up the source info

        int d_cw_start = 4 * (patch->n_sources * exposure + g);

        ConvolveOne()

    }
}


// This function sets up the gaussians for a single source and a single 
//


void ConvolveOne(int band, Patch * patch, Source *galaxy, PSFSourceGaussian *psfgauss,
              int d_cw_start, int cr_start) {


    // These are the inputs (plus the patch)
    //Source *galaxy = sources+g;
    //PSFSourceGaussian *psfgauss = patch->psfgauss+psfgauss_start + p;
    //int d_cw_start = 4 * (patch->n_sources * exposure + g);
    //int cr_start = 2 * (patch->n_sources * exposure + g);
    //------------------------

    PixGaussian    sersicgauss;    // This is where we'll queue up the source info

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

    // Source dependent coordinate reference values
    float crpix[2], crval[2];
    crpix[0] = patch->crpix[cr_start]; crpix[1] = patch->crpix[cr_start+1];
    crval[0] = patch->crval[cr_start]; crval[1] = patch->crval[cr_start+1];

    float smean[2];
    smean[0] = galaxy->ra  - crval[0];
    smean[1] = galaxy->dec - crval[1];
    sersicgauss.CW = matrix22(patch->CW+d_cw_start);
    Av(sersicgauss.CW, smean); //multiplies CW (2x2) by smean (2x1) and stores result in smean.

    int s = psfgauss->sersic_radius_bin;
    sersicgauss.xcen = smean[0] + crpix[0];
    sersicgauss.ycen = smean[1] + crpix[1];
    sersicgauss.covar = patch->rad2[s];
    sersicgauss.amp   = galaxy->mixture_amplitudes[s];
    sersicgauss.da_dn = galaxy->damplitude_dnsersic[s];
    sersicgauss.da_dr = galaxy->damplitude_drh[s] ;
    sersicgauss.flux = galaxy->fluxes[band];         //pull the correct flux from the multiband array
    // G is the conversion of flux units to image counts
    sersicgauss.G = G;
    // T is a unit circle, stretched by q, rotated by PA, and then distorted to the pixel scale
    sersicgauss.T = D * R * S;
    // And now we have the derivatives of T wrt q and PA.
    sersicgauss.dT_dq  = D * R * dS_dq;
    sersicgauss.dT_dpa = D * dR_dpa * S;

    GetGaussianAndJacobian(sersicgauss, *psfgauss, imageGauss[tid]);
        // g * n_psf_per_source + p]);
}



void EvaluateProposal(long _patch, long _proposal,void *pchi2, void *pdchi2_dp) {

    // We will use a block of shared memory
    // extern __shared__ char shared[];

    // Get the patch set up
    Patch *patch = (Patch *)_patch;

    // The Proposal is a vector of Sources[n_sources]
    Source *sources = (Source *)_proposal;

    int n_gal_gauss;
    int band_N;
    int band_start;
    int n_gauss_total;
    int n_sources = patch->n_sources;
    float dchi2_dp[NPARAMS];   // This holds the derivatives for one galaxy

    double pchi2[N_bands];
    float pdchi2_dp[N_bands, NPARAMS*MAXSOURCES];


    // Loop over bands
    for (int b = 0; b < N_bands; b++) {
        int THISBAND = b;

        n_gal_gauss = patch->n_psf_per_source[THISBAND];
        band_N = patch->band_N[THISBAND];
        band_start = patch->band_start[THISBAND];
        n_gauss_total = n_sources*n_gal_gauss;
        // imageGauss = (ImageGaussian *) (shared);

        // Loop over exposures in this band
        for (int e = 0; e < band_N; e++) {
            int exposure = band_start + e;

            // Do the convolution
            Convolve(b, patch, sources, imageGauss);

            // Compute model for each pixel
            for (int p = 0 ; p < patch->exposure_N[exposure]; p ++) {
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
                double chi2 = ((double) residual - (double) data);
                chi2 *= ((double) residual + (double) data);
                pchi2[THISBAND] += chi2;
                residual *= ierr;   // We want res*ierr^2 for the derivatives

                // Now we loop over Sources and compute the derivatives for each
                for (int gal = 0; gal < n_sources; gal++) {
                    for (int j=0; j<NPARAMS; j++) dchi2_dp[j]=0.0f;
                    ComputeGaussianDerivative(xp, yp, residual,  //1.
                            imageGauss+gal*n_gal_gauss, dchi2_dp, n_gal_gauss);

                    // accum[accumnum].SumDChi2dp(dchi2_dp, gal);
                    // FIXME: this is not right....
                    pdchi2_dp[THISBAND, gal*NPARAMS] += dchi2_dp;

            }
        }
            // Store the result
            Response *r = (Response *)pdchi2_dp;
            accum[0].store((double *) pchi2 + THISBAND, (float *) &(r[THISBAND].dchi2_dparam), n_sources);


    }

}

PYBIND11_MODULE(compute, m) {
    m.def("EvaluateProposal", &EvaluateProposal);
}

