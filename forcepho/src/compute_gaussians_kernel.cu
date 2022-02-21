/* compute_gaussian_kernels.cu

This is the code to compute a Gaussian mixture likelihood and derivative
on the GPU.  Top-level code view:

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


/// This function takes all of the sources for this exposure and
/// convolves each with the PSFGaussians for all radii, producing a long
/// set of ImageGaussians that live in the shared memory.


__device__ void CreateImageGaussians(Patch * patch, Source * sources, int exposure, ImageGaussian * imageGauss) {

    // We're going to store some values common to the exposure in shared memory
    __shared__ int band, psfgauss_start, n_psf_per_source, n_gal_gauss;
    __shared__ float G; //, crpix[2], crval[2];

    // Load the shared values
    if ( threadIdx.x == 0 ){
        band = blockIdx.x;   // This block is doing one band
        psfgauss_start = patch->psfgauss_start[exposure];
        G = patch->G[exposure];

        //crpix[0] = patch->crpix[2*exposure]; crpix[1] = patch->crpix[2*exposure + 1];
        //crval[0] = patch->crval[2*exposure]; crval[1] = patch->crval[2*exposure + 1];

        n_psf_per_source = patch->n_psf_per_source[band]; //constant per band.
        n_gal_gauss = patch->n_sources * n_psf_per_source;
        // OPTION: Consider use of __constant__ variables
    }

    __syncthreads();

    // And now we're ready for the main loop.
    // Each thread will work on one ImageGaussian, which means one PSF component
    // and one source radius for one galaxy/source.

    for (int tid = threadIdx.x; tid < n_gal_gauss; tid += blockDim.x) {
        // Unpack the source and gaussian.
        int g = tid / n_psf_per_source;       // Source number
        int p = tid - g * n_psf_per_source;   // Gaussian number

        Source *galaxy = sources+g;
        PSFSourceGaussian *psfgauss = patch->psfgauss+psfgauss_start + p;
        PixGaussian    sersicgauss;    // This is where we'll queue up the source info

        // Source dependent coordinate reference values
        float crpix[2], crval[2];
        int cr_start = 2 * (patch->n_sources * exposure + g);
        crpix[0] = patch->crpix[cr_start]; crpix[1] = patch->crpix[cr_start+1];
        crval[0] = patch->crval[cr_start]; crval[1] = patch->crval[cr_start+1];
        int d_cw_start = 4 * (patch->n_sources * exposure + g);
        float smean[2];
        smean[0] = galaxy->ra  - crval[0];
        smean[1] = galaxy->dec - crval[1];

        // Do the setup of the transformations
        //Get the transformation matrix and other conversions
        matrix22 D, R, S;

        D = matrix22(patch->D+d_cw_start);
        R.rot(galaxy->pa);
        S.scale(galaxy->q);

        //And its derivatives with respect to scene parameters
        matrix22 dS_dq, dR_dpa;
        dS_dq.scale_matrix_deriv(galaxy->q);
        dR_dpa.rotation_matrix_deriv(galaxy->pa);

        // Start filling sersicgauss
        // G is the conversion of flux units to image counts
        sersicgauss.G = G;
        // T is a unit circle, stretched by q, rotated by PA, and then distorted to the pixel scale
        sersicgauss.T = D * R * S;
        // And now we have the derivatives of T wrt q and PA.
        sersicgauss.dT_dq  = D * R * dS_dq;
        sersicgauss.dT_dpa = D * dR_dpa * S;
        // And here is the dpix/dsky matrix and the source center in pixels
        sersicgauss.CW = matrix22(patch->CW+d_cw_start);
        Av(sersicgauss.CW, smean); //multiplies CW (2x2) by smean (2x1) and stores result in smean.
        sersicgauss.xcen = smean[0] + crpix[0];
        sersicgauss.ycen = smean[1] + crpix[1];

        int s = psfgauss->sersic_radius_bin;
        sersicgauss.covar = patch->rad2[s];
        sersicgauss.amp   = galaxy->mixture_amplitudes[s];
        sersicgauss.da_dn = galaxy->damplitude_dnsersic[s];
        sersicgauss.da_dr = galaxy->damplitude_drh[s] ;
        sersicgauss.flux = galaxy->fluxes[band];         //pull the correct flux from the multiband array

        GetGaussianAndJacobian(sersicgauss, *psfgauss, imageGauss[tid]);
            // g * n_psf_per_source + p]);
    }
}



// ===================== Helper class for accumulating the results ========

class Accumulator {
  public:
    double chi2;
    float dchi2_dp[NPARAMS*MAXSOURCES];
        //OPTION: Figure out how to make this not compile time.

    __device__ Accumulator() {
    }
    __device__ ~Accumulator() { }

    __device__ void zero() {
        if (threadIdx.x==0) chi2 = 0.0;
        for (int j=threadIdx.x; j<NPARAMS*MAXSOURCES; j+=blockDim.x) dchi2_dp[j] = 0.0f;
        __syncthreads();
    }

    #define FULL_MASK 0xffffffff
    __device__ void warpReduceSum(float *answer, float input) {
        input += __shfl_down_sync(FULL_MASK, input, 16);
        input += __shfl_down_sync(FULL_MASK, input,  8);
        input += __shfl_down_sync(FULL_MASK, input,  4);
        input += __shfl_down_sync(FULL_MASK, input,  2);
        input += __shfl_down_sync(FULL_MASK, input,  1);
        // threadIdx.x % 32 == 0
        if ((threadIdx.x&31) == 0) atomicAdd(answer, input);
    }

    __device__ void warpReduceSumDbl(double *answer, double input) {
        input += __shfl_down_sync(FULL_MASK, input, 16);
        input += __shfl_down_sync(FULL_MASK, input,  8);
        input += __shfl_down_sync(FULL_MASK, input,  4);
        input += __shfl_down_sync(FULL_MASK, input,  2);
        input += __shfl_down_sync(FULL_MASK, input,  1);
        // threadIdx.x % 32 == 0
        if ((threadIdx.x&31) == 0) atomicAdd(answer, input);
    }

    // Could put the Reduction code in here
    __device__ void SumChi2(double _chi2) {
        warpReduceSumDbl(&chi2, _chi2);
    }
    __device__ void SumDChi2dp(float *_dchi2_dp, int gal) {
        for (int j=0; j<NPARAMS; j++)
            warpReduceSum(dchi2_dp+NPARAMS*gal+j, _dchi2_dp[j]);
    }

    /// This copies this Accumulator into another memory buffer
    __device__ inline void store(double *pchi2, float *pdchi2_dp, int n_sources) {
        if (threadIdx.x==0) *pchi2 = chi2;
        for (int j=threadIdx.x; j<n_sources*NPARAMS; j+=blockDim.x)
            pdchi2_dp[j] = dchi2_dp[j];
        __syncthreads();
    }

    __device__ inline void addto(Accumulator &A, int n_sources) {
        if (threadIdx.x==0) chi2 += A.chi2;
        for (int j=threadIdx.x; j<n_sources*NPARAMS; j+=blockDim.x)
            dchi2_dp[j] += A.dchi2_dp[j];
        __syncthreads();
    }

    __device__ void coadd_and_sync(Accumulator *A, int nAcc, int n_sources) {
        for (int n=1; n<nAcc; n++) addto(A[n], n_sources);
            __syncthreads();
    }
};

// ================= Primary Proposal Kernel ========================


/// We are being handed pointers to a Patch structure, a Proposal structure,

/// a scalar chi2 response, and a vector dchi2_dp response.
/// The proposal is a pointer to Source[n_sources] sources.
/// The response is a pointer to [band][MaxSource] Responses.

#define THISBAND blockIdx.x
// Creating a more interpretable shorthand for this

extern "C" {
__global__ void EvaluateProposal(void *_patch, void *_proposal,
                                 void *pchi2, void *pdchi2_dp) {
    // We will use a block of shared memory
    extern __shared__ char shared[];

    // Get the patch set up
    Patch *patch = (Patch *)_patch;

    // The Proposal is a vector of Sources[n_sources]
    Source *sources = (Source *)_proposal;

    // Allocate the ImageGaussians for this band (same number for all exposures)
    __shared__ int n_gal_gauss;   // Number of image gaussians per galaxy
    __shared__ int band_N;   // The number of exposures in this band
    __shared__ int band_start;   // The starting exposures in this band
    __shared__ int n_sources;   // The number of sources
    __shared__ int n_gauss_total;   // Number of image gaussians for all sources
    __shared__ ImageGaussian *imageGauss; // [source][gauss]
    __shared__ Accumulator *accum;   // [NUMACCUMS]

    if (threadIdx.x==0) {
        int shared_counter = 0;

        n_gal_gauss = patch->n_psf_per_source[THISBAND];
        band_N = patch->band_N[THISBAND];
        band_start = patch->band_start[THISBAND];
        n_sources = patch->n_sources;
        n_gauss_total = n_sources*n_gal_gauss;
        accum = (Accumulator *) shared;
        shared_counter += NUMACCUMS*sizeof(Accumulator);
        imageGauss = (ImageGaussian *) (shared + shared_counter);
    }
    __syncthreads();   // Have to get this malloc done

    for (int j=0; j<NUMACCUMS; j++) accum[j].zero();
    float dchi2_dp[NPARAMS];   // This holds the derivatives for one galaxy

    // Now figure out which one this thread should use
    int threads_per_accum = ceilf(blockDim.x/warpSize/NUMACCUMS)*warpSize;
    int accumnum = threadIdx.x / threads_per_accum;  // We are accumulating each warp separately.

    // Loop over Exposures
    for (int e = 0; e < band_N; e++) {
        int exposure = band_start + e;

        CreateImageGaussians(patch, sources, exposure, imageGauss);

        __syncthreads();

        for (int p = threadIdx.x ; p < patch->exposure_N[exposure]; p += blockDim.x) {
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
            // below computes (r^2 - d^2) / sigma^2 in an attempt to decrease the size of the
            // residual and avoid loss of significance
            // promoting to double on the right hand side only modestly increases the precision,
            // but also does not measureably impact the timing
            double chi2 = ((double) residual - (double) data);
            chi2 *= ((double) residual + (double) data);
            // chi2 -= 36.9;
            accum[accumnum].SumChi2(chi2);
            residual *= ierr;   // We want res*ierr^2 for the derivatives

            // Now we loop over Sources and compute the derivatives for each
            for (int gal = 0; gal < n_sources; gal++) {
                for (int j=0; j<NPARAMS; j++) dchi2_dp[j]=0.0f;
                ComputeGaussianDerivative(xp, yp, residual,  //1.
                        imageGauss+gal*n_gal_gauss, dchi2_dp, n_gal_gauss);

                //if(gal == 0)
                //    patch->residual[pix] = dchi2_dp[1];

                accum[accumnum].SumDChi2dp(dchi2_dp, gal);

            }
        }

    __syncthreads();
    }

    // Now we're done with all exposures, but we need to sum the Accumulators
    // over all warps.
    accum[0].coadd_and_sync(accum, NUMACCUMS, n_sources);
    Response *r = (Response *)pdchi2_dp;
    accum[0].store((double *) pchi2 + THISBAND, (float *) &(r[THISBAND].dchi2_dparam), n_sources);
    return;
}

}  // extern "C"
