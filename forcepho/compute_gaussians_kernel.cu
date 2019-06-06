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
#include "patch.cu"
#include "proposal.cu"

// =====================  ImageGaussian class =============================

class ImageGaussian {
  public:
    // 6 Gaussian parameters
	float amp;
	float xcen; 
	float ycen;
	float fxx; 
	float fyy;
	float fxy; 
	
    // 15 Jacobian elements (Image -> Sky)
    float dA_dFlux;
    float dx_dAlpha;
    float dy_dAlpha;
    float dx_dDelta;
    float dy_dDelta;
    float dA_dQ;
    float dFxx_dQ;
    float dFyy_dQ;
    float dFxy_dQ;
    float dA_dPA;
    float dFxx_dPA;
    float dFyy_dPA;
    float dFxy_dPA;
    float dA_dSersic;
    float dA_drh;
};

// ======================  Code to Evaluate the Gaussians =========================

/// Compute the Model for one pixel from all galaxies; return the residual image
/// The one pixel is specified with xp, yp, data.
/// We have to enter a pointer to the whole list of ImageGaussians.

// TODO: n_gauss_total is a shared scalar in the calling function, but not here.
// Can we avoid the thread-based storage?

__device__ PixFloat ComputeResidualImage(float xp, float yp, PixFloat data, ImageGaussian * imageGauss, int n_gauss_total)
{
	PixFloat residual = data;
	
	//loop over all image gaussians g for all galaxies. 
	for (int i = 0; i < n_gauss_total; i ++){
		ImageGaussian *g = imageGauss+i;
        // TODO: Could say imageGauss++;
		float dx = xp - g->xcen; 
		float dy = yp - g->ycen; 
		float vx = g->fxx * dx + g->fxy * dy;
		float vy = g->fyy * dy + g->fxy * dx;
		float exparg = dx*vx+dy*vy;
		if (exparg>MAX_EXP_ARG) continue;
		float Gp = exp(-0.5 * exparg);

		// Here are the second-order corrections to the pixel integral
		float H = 1.0 + (vx*vx + vy*vy - g->fxx - g->fyy) / 24.0; 
		float C = g->amp * Gp * H; //count in this pixel. 
		
		residual -= C; 
	}
	return residual;
}

// ====================  Code to Compute the Derivatives =====================

/// Compute the likelihood derivative for one pixel and one galaxy.
/// The one pixel is specified with xp, yp, and the residual*ierr^2.
/// We have to enter a pointer to the Gaussians for this one galaxy.
/// And we supply a pointer where we accumulate the derivatives.

// TODO: n_gal_gauss is a shared scalar in the calling function, but not here.
// Can we avoid the thread-based storage?

__device__ void ComputeGaussianDerivative(float xp, float yp, float residual_ierr2, 
            ImageGaussian *gaussian, float * dchi2_dp, int n_gal_gauss) 
{
    // Loop over all gaussians in this galaxy. 
	for (int gauss = 0; gauss<n_gal_gauss; gauss++) {   
		ImageGaussian *g = gaussian+gauss;
        // TODO: could just say gaussian++;
	
		float dx = xp - g->xcen; 
		float dy = yp - g->ycen; 
		float vx = g->fxx * dx + g->fxy * dy;
		float vy = g->fyy * dy + g->fxy * dx;
		float Gp = exp(-0.5 * (dx*vx + dy*vy));
		float H = 1.0 + (vx*vx + vy*vy - g->fxx - g->fyy) *(1.0/24.0); 
	
        // Old code: this had divisions
		// float C = residual_ierr2 * g->amp * Gp * H;   
	    // float dC_dA   = C / g->amp;
	    // float c_h = C / H;

		float dC_dA = residual_ierr2 * Gp;
        float c_h = dC_dA * g->amp;
        dC_dA *= H;
	    float C   = dC_dA * H;
	    float dC_dx   = C*vx;
	    float dC_dy   = C*vy;
	    float dC_dfx  = -0.5*C*dx*dx;
	    float dC_dfy  = -0.5*C*dy*dy;
	    float dC_dfxy = -1.0*C*dx*dy;
	
	    dC_dx    -= c_h * (g->fxx*vx + g->fxy*vy) * (1.0/12.0);
	    dC_dy    -= c_h * (g->fyy*vy + g->fxy*vx) * (1.0/12.0);
	    dC_dfx   -= c_h * (1.0 - 2.0*dx*vx) * (1.0/24.0);
	    dC_dfy   -= c_h * (1.0 - 2.0*dy*vy) * (1.0/24.0);
	    dC_dfxy  += c_h * (dy*vx + dx*vy) * (1.0/12.0);
			 
	    //Multiply by Jacobian and add to dchi2_dp	
		dchi2_dp[0] += g->dA_dFlux * dC_dA ; 
		dchi2_dp[1] += g->dx_dAlpha * dC_dx + g->dy_dAlpha * dC_dy;
		dchi2_dp[2] += g->dx_dDelta * dC_dx + g->dy_dDelta * dC_dy;
		dchi2_dp[3] += g->dA_dQ  * dC_dA + dC_dfx * g->dFxx_dQ + dC_dfxy * g->dFxy_dQ + dC_dfy * g->dFyy_dQ;
		dchi2_dp[4] += g->dA_dPA * dC_dA + dC_dfx * g->dFxx_dPA + dC_dfxy * g->dFxy_dPA + dC_dfy * g->dFyy_dPA;
		dchi2_dp[5] += g->dA_dSersic * dC_dA;
		dchi2_dp[6] += g->dA_drh * dC_dA;	
	}
}


// =================== Code to prepare the Gaussians =======================

/// This holds the information on a single source Gaussian, along with 
/// info needed to compute derivatives post-convolution.

typedef struct { 
    // 6 Gaussian parameters for sersic profile 
	float amp;
	float xcen;
	float ycen;
	float covar; //diagonal element of sersic profile covariance. covariance = covar * I. 
	matrix22 scovar_im; 
	
	//some distortions and astrometry specific to ??source??. 
	float flux; 
	float G; 
	float da_dn;
	float da_dr;
	matrix22 CW; 
	matrix22 T; 
	matrix22 dT_dq;
	matrix22 dT_dpa;
} PixGaussian; 

/// This function takes a source Gaussian and a PSF Gaussian and 
/// performs the convolution, creating the ImageGaussian that contains
/// the on-image gaussian plus the Jacobian to convert derivatives into
/// the on-sky parameters.

__device__ void  GetGaussianAndJacobian(PixGaussian & sersicgauss, PSFSourceGaussian & psfgauss, ImageGaussian & gauss){

	sersicgauss.scovar_im = sersicgauss.covar * AAt(sersicgauss.T); 
		
	matrix22 covar = sersicgauss.scovar_im + matrix22(psfgauss.Cxx, psfgauss.Cxy, psfgauss.Cxy, psfgauss.Cyy); 
	matrix22 f = covar.inv(); 
	float detF = f.det(); 
	
	gauss.fxx = f.v11; 
	gauss.fxy = f.v21; 
	gauss.fyy = f.v22; 
	
	gauss.xcen = sersicgauss.xcen + psfgauss.xcen; 
	gauss.ycen = sersicgauss.ycen + psfgauss.ycen; 
	
	gauss.amp = sersicgauss.flux * sersicgauss.G * sersicgauss.amp * psfgauss.amp * sqrt(detF) * (1.0/(2.0*M_PI)) ;

	//now get derivatives of F
	matrix22 dSigma_dq  = sersicgauss.covar * (sersicgauss.T * sersicgauss.dT_dq.T()  + sersicgauss.dT_dq  * sersicgauss.T.T() ) ; 
	matrix22 dSigma_dpa = sersicgauss.covar * (sersicgauss.T * sersicgauss.dT_dpa.T() + sersicgauss.dT_dpa * sersicgauss.T.T() ) ; 
    // TODO: This math of A B^T + B A^T could be simplified:
    // it's the symmetrization of a 2x2 matrix
	
	matrix22 dF_dq      = -1.0 * ABA (f, dSigma_dq);  // F *  dSigma_dq * F
	matrix22 dF_dpa     = -1.0 * ABA (f, dSigma_dpa); // F * dSigma_dpa * F
	
	float ddetF_dq   = detF *  (covar * dF_dq).trace(); 
	float ddetF_dpa  = detF * (covar * dF_dpa).trace(); 
	
	// Now get derivatives with respect to sky parameters
    gauss.dA_dQ      = gauss.amp /(2.0*detF) * ddetF_dq;  
    gauss.dA_dPA     = gauss.amp /(2.0*detF) * ddetF_dpa;  
    // TODO: Why do we multiply and then divide by detF?
    gauss.dA_dFlux   = gauss.amp / sersicgauss.flux; 
    gauss.dA_dSersic = gauss.amp / sersicgauss.amp * sersicgauss.da_dn;
    gauss.dA_drh     = gauss.amp / sersicgauss.amp * sersicgauss.da_dr;
    // TODO: Some opportunity in the above to avoid some divisions.
	
	gauss.dx_dAlpha = sersicgauss.CW.v11; 
	gauss.dy_dAlpha = sersicgauss.CW.v21; 
	
	gauss.dx_dDelta = sersicgauss.CW.v12;
	gauss.dy_dDelta = sersicgauss.CW.v22; 
	
	gauss.dFxx_dQ = dF_dq.v11;
	gauss.dFyy_dQ = dF_dq.v22;
	gauss.dFxy_dQ = dF_dq.v21; 

	gauss.dFxx_dPA = dF_dpa.v11;
	gauss.dFyy_dPA = dF_dpa.v22;
	gauss.dFxy_dPA = dF_dpa.v21; 
}

/// This function takes all of the sources for this exposure and 
/// convolves each with the PSFGaussians for all radii, producing a long
/// set of ImageGaussians that live in the shared memory.

__device__ void CreateImageGaussians(Patch * patch, Source * sources, int exposure, ImageGaussian * imageGauss) {
	
    // We're going to store some values common to the exposure in shared memory
	__shared__ int band, psfgauss_start, n_psf_per_source, n_gal_gauss; 
	__shared__ float G, crpix[2], crval[2]; 
	
    // Load the shared values
	if ( threadIdx.x == 0 ){
	    band = blockIdx.x;   // This block is doing one band
		psfgauss_start = patch->psfgauss_start[exposure];
		G = patch->G[exposure]; 
	
		crpix[0] = patch->crpix[2*exposure]; crpix[1] = patch->crpix[2*exposure + 1];
		crval[0] = patch->crval[2*exposure]; crval[1] = patch->crval[2*exposure + 1];
	
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
		PixGaussian	sersicgauss;    // This is where we'll queue up the source info
	
	    // Do the setup of the transformations		
		//Get the transformation matrix and other conversions
		matrix22 D, R, S; 
		
		int d_cw_start = 4 * (patch->n_sources * exposure + g); 
		D  = matrix22(patch->D+d_cw_start);		
		R.rot(galaxy->pa); 
		S.scale(galaxy->q); 
	
		//And its derivatives with respect to scene parameters
		matrix22 dS_dq, dR_dpa;
		dS_dq.scale_matrix_deriv(galaxy->q);
		dR_dpa.rotation_matrix_deriv(galaxy->pa);
				
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
		sersicgauss.flux = galaxy->fluxes[band]; 		//pull the correct flux from the multiband array
		sersicgauss.G = G; 
		sersicgauss.T = D * R * S; 
		sersicgauss.dT_dq  = D * R * dS_dq; 
		sersicgauss.dT_dpa = D * dR_dpa * S; 

    	GetGaussianAndJacobian(sersicgauss, *psfgauss, imageGauss[g * n_psf_per_source + p]);
        // TODO: Wasn't this argument just tid?
	}
}
	
	

// ===================== Helper class for accumulating the results ========

class Accumulator {
  public:
    float chi2;
    float dchi2_dp[NPARAMS*MAXSOURCES]; 
        //OPTION: Figure out how to make this not compile time.

    __device__ Accumulator() {
    }
    __device__ ~Accumulator() { }

    __device__ void zero() {
        if (threadIdx.x==0) chi2 = 0.0;
        for (int j=threadIdx.x; j<NPARAMS*MAXSOURCES; j+=blockDim.x) dchi2_dp[j] = 0.0;
        __syncthreads();
    }

	#define FULL_MASK 0xffffffff
    __device__ void warpReduceSum(float *answer, float input) {
        input += __shfl_down_sync(FULL_MASK, input, 16);
        input += __shfl_down_sync(FULL_MASK, input,  8);
        input += __shfl_down_sync(FULL_MASK, input,  4);
        input += __shfl_down_sync(FULL_MASK, input,  2);
        input += __shfl_down_sync(FULL_MASK, input,  1);
        if (threadIdx.x&31==0) atomicAdd(answer, input);
    }
    
    // Could put the Reduction code in here
    __device__ void SumChi2(float _chi2) {
        warpReduceSum(&chi2, _chi2);
    }
    __device__ void SumDChi2dp(float *_dchi2_dp, int gal) { 
        for (int j=0; j<NPARAMS; j++) 
            warpReduceSum(dchi2_dp+NPARAMS*gal+j, _dchi2_dp[j]); 
    }

    /// This copies this Accumulator into another memory buffer
    __device__ inline void store(float *pchi2, float *pdchi2_dp, int n_active) {
        if (threadIdx.x==0) *pchi2 = chi2;
        for (int j=threadIdx.x; j<n_active*NPARAMS; j+=blockDim.x)
            pdchi2_dp[j] = dchi2_dp[j];
        __syncthreads();
    }

    __device__ inline void addto(Accumulator &A, int n_active) {
        if (threadIdx.x==0) chi2 += A.chi2;
        for (int j=threadIdx.x; j<n_active*NPARAMS; j+=blockDim.x)
            dchi2_dp[j] += A.dchi2_dp[j];
    }

    __device__ void coadd_and_sync(Accumulator *A, int nAcc, int n_active) {
        for (int n=1; n<nAcc; n++) addto(A[n], n_active);
        __syncthreads();
    }
};

// ================= Primary Proposal Kernel ========================

/// We are being handed pointers to a Patch structure, a Proposal structure,
/// a scalar chi2 response, and a vector dchi2_dp response.
/// The proposal is a pointer to Source[n_active] sources.
/// The response is a pointer to [band][MaxSource] Responses.

#define THISBAND blockIdx.x
// Creating a more interpretable shorthand for this

extern "C" {
__global__ void EvaluateProposal(void *_patch, void *_proposal, 
                                 void *pchi2, void *pdchi2_dp) {
    // We will use a block of shared memory
    extern __shared__ char shared[];
    int shared_counter = 0;
    //char *shared = (char *)_shared;

    // Get the patch set up
    Patch *patch = (Patch *)_patch;  

    // The Proposal is a vector of Sources[n_active]
    Source *sources = (Source *)_proposal;
	
    // Now figure out which Accumulator this thread should use
    int threads_per_accum = ceilf(blockDim.x/warpSize/NUMACCUMS)*warpSize;
    int accumnum = threadIdx.x / threads_per_accum;  // We are accumulating each warp separately. 
	
    // Allocate the ImageGaussians for this band (same number for all exposures)
    __shared__ int n_gal_gauss;   // Number of image gaussians per galaxy
    __shared__ int band_N;   // The number of exposures in this band
    __shared__ int band_start;   // The starting exposures in this band
    __shared__ int n_sources;   // The number of sources
    __shared__ int n_gauss_total;   // Number of image gaussians for all sources
    __shared__ ImageGaussian *imageGauss; // [source][gauss]
    __shared__ Accumulator *accum;   // [NUMACCUMS]
    
    if (threadIdx.x==0) {
        n_gal_gauss = patch->n_psf_per_source[THISBAND];
        band_N = patch->band_N[THISBAND];
        band_start = patch->band_start[THISBAND];
        n_sources = patch->n_sources;
        n_gauss_total = n_sources*n_gal_gauss;
        accum = (Accumulator *) shared;
        //shared += sizeof(Accumulator)*NUMACCUMS;
        shared_counter += NUMACCUMS*sizeof(Accumulator);
        imageGauss = (ImageGaussian *) (shared + shared_counter);
    }
    __syncthreads();   // Have to get this malloc done

    for (int j=0; j<NUMACCUMS; j++) accum[j].zero();
    float dchi2_dp[NPARAMS];   // This holds the derivatives for one galaxy

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
		    float chi2 = residual*residual;
		    accum[accumnum].SumChi2(chi2);
            residual *= ierr;   // We want res*ierr^2 for the derivatives
	    
		    // Now we loop over Sources and compute the derivatives for each
		    for (int gal = 0; gal < n_sources; gal++) {
				for (int j=0; j<NPARAMS; j++) dchi2_dp[j]=0.0;
				ComputeGaussianDerivative(xp, yp, residual, 
                        imageGauss+gal*n_gal_gauss, dchi2_dp, n_gal_gauss);  

				accum[accumnum].SumDChi2dp(dchi2_dp, gal);

		    }
		}

	__syncthreads();
    }
    // Now we're done with all exposures, but we need to sum the Accumulators
    // over all warps.
    accum[0].coadd_and_sync(accum, NUMACCUMS, n_sources);
    Response *r = (Response *)pdchi2_dp;
    accum[0].store((float *)pchi2, (float *) &(r[THISBAND].dchi2_dparam), n_sources);
    return;
}

}  // extern "C"
