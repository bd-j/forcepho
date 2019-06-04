/* 
NWarps = BlockSize/WarpSize(=32)

Create chi^2 and d(chi2/dparam) in shared memory and zero them.
    chi^2[NWarps]
    dchi_dp[NWarps][NActiveGalaxy]

For each exposure:

    Create on-image Gaussians from on-sky parameters, put in shared memory
        ImageGaussian[NGalaxy*GaussianPerGalaxy]

    For one pixel per thread (taking BlockSize steps):

        Load Image Data
        Loop over all ImageGaussians:
            Evaluate Gaussians to create Residual image, save in register

        Compute local_chi2 from residual image for this pixel
        Reduce local_chi2 over warp; atomic_add result to shared mem

        Loop over Active Galaxy:
            Loop over Gaussian in this Galaxy:
                Compute local_dchi_dp and accumulate
            Reduce local_dchi_dp over warp and atomic_add to shared dchi_dp for galaxy
            
*/

/*
class Patch {
    // Exposure data[], ierr[], xpix[], ypix[], astrometry
    // List of FixedGalaxy
    // Number of SkyGalaxy
    int nActiveGals; 
    int nFixedGals;
    SkyGalaxy * FixedGals;
    //..list of fixed galaxies? 
    PixFloat * data;
    PixFloat * ierr;
    PixFloat * xpix; 
    PixFloat * ypix; 
    //..astrometry? 
    PSFGaussian * psfgauss;
    int * nPSFGauss;   // Indexed by exposure
    int * startPSFGauss;   // Indexed by exposure
};

class SkyGalaxy {
    // On-sky galaxy parameters
    // flux: total flux
    // ra: right ascension (degrees)
    // dec: declination (degrees)
    // q, pa: axis ratio squared and position angle
    // n: sersic index
    // r: half-light radius (arcsec)
    float flux; 
    float ra; 
    float dec;
    float q; 
    float pa; 
    float n;
    float r; 
};
class Proposal {
    SkyGalaxy * ActiveGals;     // List of SkyGalaxy, input for this HMC likelihood
};

typedef struct{
    int nGauss; 
    ImageGaussian * gaussians; //List of ImageGaussians
    // TODO: May not need this to be explicit
} Galaxy;
*/  
//=================== ABOVE THIS LINE IS DEPRECATED ============

typedef float PixFloat;
#define NPARAM 7    // Number of Parameters per Galaxy, in one band 
#define MAXACTIVE 30    // Max number of active Galaxies in a patch

typedef struct {
    // 6 Gaussian parameters
    float amp;
    float xcen; 
    float ycen;
    float Cxx; 
    float Cyy;
    float Cxy; 
    // TODO: Consider whether a dummy float will help with shared memory bank constraints
} PixGaussian;

typedef struct {
    // 6 Gaussian parameters
    float amp;
    float xcen; 
    float ycen;
    float fxx; 
    float fyy;
    float fxy; 
    // TODO: Consider whether a dummy float will help with shared memory bank constraints
} ImageGaussian;

typedef struct {
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
} ImageGaussianJacobian;


#define MAX_EXP_ARG 36.0

__device__ PixFloat ComputeResidualImage(float xp, float yp, PixFloat data, Patch patch, Galaxy galaxy); //NAM do we need patch, galaxy? 
{
    PixFloat residual = data;
    
    // TODO: Need to loop over Active and Fixed galaxies
    //loop over all image gaussians g. 
    for (int i = 0; i < galaxy.nGauss; i ++){ //NAM TODO nGauss may be derived from Patch class properties. 
        ImageGaussian g = galaxy.Gaussians[i]
        float dx = xp - g.xcen; 
        float dy = yp - g.ycen; 
        float vx = g.fxx * dx + g.fxy * dy;
        float vy = g.fyy * dy + g.fxy * dx;
        float exparg = dx*vx+dy*vy;
        if (exparg>MAX_EXP_ARG) continue;
        float Gp = exp(-0.5 * exparg);

        // Here are the second-order corrections to the pixel integral
        float H = 1.0 + (vx*vx + vy*vy - g.fxx - g.fyy) / 24.0; 
        float C = g.amp * Gp * H; //count in this pixel. 
        
        residual -= C; 
    }
    return residual;
}

__device__ void ComputeGaussianDerivative(pix, xp, yp, residual, gal, gauss, float * dchi2_dp) //NAM why are we passing in residual? it's been accumulated over ImageGaussians earlier... we need to repeat work here to get isolated Image Gaussian
{
    float dx = xp - gauss.xcen; 
    float dy = yp - gauss.ycen; 
    float vx = gauss.fxx * dx + gauss.fxy * dy;
    float vy = gauss.fyy * dy + gauss.fxy * dx;
    float Gp = exp(-0.5 * (dx*vx + dy*vy));
    float H = 1.0; 
    float root_det = 1.0;             
    
    H = 1.0 + (vx*vx + vy*vy - gauss.fxx - gauss.fyy) / 24.0; 
    float C = gauss.amp * Gp * H * root_det; //count in this pixel. 
    
    float dC_dA = C / gauss.amp;
    float dC_dx = C*vx;
    float dC_dy = C*vy;
    float dC_dfx = -0.5*C*dx*dx;
    float dC_dfy = -0.5*C*dy*dy;
    float dC_dfxy = -1.0*C*dx*dy;
    
    float c_h = C / H;
    dC_dx -= c_h * (g.fxx*vx + g.fxy*vy) / 12.0;
    dC_dy -= c_h * (g.fyy*vy + g.fxy*vx) / 12.0;
    dC_dfx -= c_h * (1.0 - 2.0*dx*vx) / 24.0;
    dC_dfy -= c_h * (1.0 - 2.0*dy*vy) / 24.0;
    dC_dfxy += c_h * (dy*vx + dx*vy) / 12.0;
    
    
    gradients += np.matmul(g.derivs, dI_dphi)
    gradients[0][:] = dC_dA[:]
    gradients[1][:] = dC_dx[:]
    gradients[2][:] = dC_dy[:]
    gradients[3][:] = dC_dfx[:]
    gradients[4][:] = dC_dfy[:]
    gradients[5][:] = dC_dfxy[:]
             
             
             

    dchi2_dpim[0] += residual * dC_dA; //NAM TODO ??  is this right? 
    dchi2_dpim[1] += residual * dC_dx;
    dchi2_dpim[2] += residual * dC_dy;
    dchi2_dpim[3] += residual * dC_dfx;
    dchi2_dpim[4] += residual * dC_dfy;
    dchi2_dpim[5] += residual * dC_dfxy;

    // TODO: Multiply by Jacobian and add to dchi2_dp
}


#define NACTIVE MAXACTIVE   // Hack for now

void warpReduceSum(float *answer, float input) {
    input += __shfl_down(input, 16);
    input += __shfl_down(input,  8);
    input += __shfl_down(input,  4);
    input += __shfl_down(input,  2);
    input += __shfl_down(input,  1);
    if (threadIdx.x&31==0) *answer = input;
}

class Accumulator {
  public:
    float chi2;
    float dchi2_dp[NPARAM*NACTIVE]; //TODO: Need to figure out how to make this not compile time.

    Accumulator() {
        chi2 = 0.0;
        for (int j=0; j<NPARAM*NACTIVE; j++) dchi2_dp[j] = 0.0;
    }
    ~Accumulator() { }
    
    // Could put the Reduction code in here
    void SumChi2(float _chi2) { warpReduceSum(&chi2, _chi2); }
    void SumDChi2dp(float *_dchi2_dp, int gal) { 
    for (int j=0; j<NPARAM; j++) 
        warpReduceSum(dchi2_dp+j+NPARAM*gal, _dchi2_dp[j]); 
    }

    /// This copies this Accumulator into another memory buffer
    inline void store(float *pchi2, float *pdchi2_dp, int nActive) {
        if (threadIdx.x==0) *pchi2 = chi2;
        for (int j=threadIdx.x; j<nActive*NPARAM; j+=BlockDim.x)
            pdchi2_dp[j] = dchi2_dp[j];
    }

    inline void addto(Accumulator &A) {
        if (threadIdx.x==0) chi2 += A.chi2;
        for (int j=threadIdx.x; j<nActive*NPARAM; j+=BlockDim.x)
            dchi2_dp[j] += A.dchi2_dp[j];
    }

    void coadd_and_sync(Accumulator *A, int nAcc) {
        for (int n=1; n<nAcc; n++) addto(A[n]);
        __syncthreads();
    }
};





__device__ void ConstructImageGaussian(int s, int p, Galaxy gal, matrix22 scovar, matrix22 pcovar, 
                                                matrix22 smean, float samp, matrix22 pmean, float pamp, float flux, ImageGaussian &gauss){
    matrix22 covar = scovar + pcovar; 
    matrix22 f = covar.inv(); 
    
    gauss.fxx = f.v11; 
    gauss.fxy = f.v21; 
    gauss.fyy = f.v22; 
    
    gauss.xcen = smean.v11 + pmean.v11; //NAM careful! 
    gauss.ycen = smean.v22 + pmean.v22; 
    
    gauss.amp = flux * samp * pamp * pow(f.det(), 0.5) / (2.0 * math.pi) ;
}

__device__ void ConstructImageJacobian(int s, int p, Galaxy gal, matrix22 scovar, matrix22 pcovar, 
                                                float samp, float pamp, float flux, float G, matrix22 T, matrix22 dT_dq, matrix22 dT_dpa, float da_dn, float da_dr, matrix22 CW, ImageGaussianJacobian &jacobian){

    //convolve the s-th Source component with the p-th PSF component.
    matrix22 scovar_im = T * scovar * T.T();
    matrix22 Sigma = scovar_im + pcovar; 
    matrix22 F = Sigma.inv(); 
    float detF = F.det(); 
    float K = flux * G * samp * pamp * pow(detF, 0.5) / (2.0 * math.pi); 
    
    //now get derivatives 
    //of F
    matrix22 dSigma_dq  = T * scovar * dT_dq.T() + dT_dq * scovar * T.T(); 
    matrix22 dSigma_dpa = T * scovar * dT_dpa.T() + dT_dpa * scovar * T.T(); 
    matrix22 dF_dq      = -F * dSigma_dq * F; 
    matrix22 dF_dpa     = -F * dSigma_dpa * F; 
    float ddetF_dq   = detF *  (Sigma * dF_dq).trace(); 
    float ddetF_dpa  = detF * (Sigma * dF_dpa).trace(); 
    
    //of Amplitude
    jacobian.dA_dQ = K / (2.0 * detF) * ddetF_dq;  
    jacobian.dA_dpA = K / (2.0 * detF) * ddetF_dpa;  
    jacobian.dA_dFlux = K / flux; 
    jacobian.dA_dSersic = K / am * da_dn;
    jacobian.dA_drh = K / am * da_dr;
    
    jacobian.dx_dAlpha = CW.v11; 
    jacobian.dy_dAlpha = CW.v21; 
    
    jacobian.dx_dDelta = CW.v12;
    jacobian.dy_dDelta = CW.v22; 
    
    jacobian.dFxx_dQ = dF_dq.v11;
    jacobian.dFyy_dQ = dF_dq.v22;
    jacobian.dFxy_dQ = dF_dq.v21; 

    jacobian.dFxx_dPA = dF_dpa.v11;
    jacobian.dFyy_dPA = dF_dpa.v22;
    jacobian.dFxy_dPA = dF_dpa.v21; 
}

__device__ void CreateImageGaussians() {
    
    int tid = threadIdx.x; 
    
    int totGals = nActiveGals+nFixedGals; 
    int tmp = (tid - p * patch->nSersicGauss * totGals); 

    int p = tid / patch->nSersicGauss * totGals; 
    int s = tmp/totGals; 
    int g = tmp%totGals; 
    
    
    
    
    
    
    
    for (int gal=0; gal<nActiveGals+nFixedGals; gal++) {
        Patch patch = ;//NAM TODO
        
        // Do the setup of the transformations        
        //Get the transformation matrix and other conversions
        // TODO: Probably not implementing the sky distortions in full
        matrix22 D = matrix22(patch.scale[0], patch.scale[1]); //diagonal 2x2 matrix. 
        matrix22 R = rot(galaxy.pa);   // TODO: fix calling syntax
        matrix22 S = scale(galaxy.q); 
        matrix22 T = D * R * S; 
        matrix22 CW = matrix22(patch.dpix_dsky[0], patch.dpix_dsky[1]);
        float G = patch.photocounts; 
        
        //And its derivatives with respect to scene parameters
        matrix22 dS_dq = scale_matrix_deriv(galaxy.q);
        matrix22 dR_dpa = rotation_matrix_deriv(galaxy.pa);
        matrix22 dT_dq = D * R * dS_dq; 
        matrix22 dT_dpa = D * dR_dpa * S;     
        
        
        for (int s=0; s<patch->nSersicGauss; s++) {
            //get source spline and derivatives
            smean = patch.sky_to_pix([source.ra, source.dec]) //NAM TODO    //these don't have to be matrix22s. just two numbers... 
                    
            matrix22 scovar = matrix22(galaxy.covariances[s], galaxy.covariances[s]) ; //diagonal elements of this gaussian's covariance matrix for sersic index s. 
            float samp = galaxy.amplitudes[s]; 
            float da_dn = galaxy.damplitude_dsersic[s];
            float da_dr = galaxy.damplitude_drh[s] ; 
            
            //pull the correct flux from the multiband array
            float flux = patch.flux[blockId.x]; //NAM TODO is this right? 
            
            for (int p=0; p<nPSFGauss; p++) {
                float pamp = patch.psf.amplitudes[p]; 
                
                //get PSF component means and covariances in the pixel space
                if (patch.psf.units[p] == 'arcsec'){
                    matrix22 pcovar = D * patch.psf.covariances[p] * D.T();
                    matrix22 pmean = D * patch.psf.means[p];  //these don't have to be matrix22s. just two numbers... 
                }
                else if (patch.psf.units == 'pixels'){
                    matrix22 pcovar = patch.psf.covariances[p]; 
                    matrix22 pmean = stamp.psf.means[p]; //these don't have to be matrix22s. just two numbers... 
                }
                
                ConstructImageGaussian(s,p,gal, T * scovar T.T(), pcovar, smean, samp, pmean, pamp, flux, imageGauss[gal*nGalGauss+s*nPSFGauss+p]);
                
                if (gal<nActiveGals) {
                    ConstructImageJacobian(s,p,gal, scovar, pcovar, samp, pamp, flux, G, T, dT_dq, dT_dpa, da_dn, da_dr, CW, imageJacob[gal*nGalGauss+s*nPSFGauss+p]);
                }
            }
        }
    }
}


// ================= Primary Proposal Kernel ========================

// Shared memory is arranged in 32 banks of 4 byte stagger
#define WARPSIZE 32

/// We are being handed pointers to a Patch structure, a Proposal structure,
/// a scalar chi2 response, and a vector dchi2_dp response
__global__ void EvaluateProposal(void *_patch, void *_proposal, 
                                 void *pchi2, void *pdchi2_dp) {
    Patch *patch = (Patch *)_patch;  // We should be given this pointer

    // The Proposal is a vector of Sources[n_active]
    Source *sources = (Source *)_proposal;

    int band = blockIdx.x;   // This block is doing one band
    int warp = threadIdx.x / WARPSIZE;  // We are accumulating in warps.

    // Create and Zero Accumulators();
    __shared__ Accumulator accum[blockDim.x/WARPSIZE]();

    // Loop over Exposures
    for (e = 0; e < patch->NumExposures[band]; e++) {
        int exposure = patch->StartExposures[band] + e;
        int nPSFGauss = patch->nPSFGauss[exposure];
        int startPSFGauss = patch->startPSFGauss[exposure];
        int nGalGauss = nPSFGauss*patch->nSersicGauss;

        __shared__ ImageGaussians imageGauss[nGalGauss*(nActiveGals+nFixedGals)];
            // Convention is Active first, then Fixed.
        __shared__ ImageGaussiansJacobians imageJacob[nGalGauss*(nActiveGals)];
            // We only need the Active galaxies
            CreateImageGaussians(patch, exposure);

        __syncthreads();

        for (p = threadIdx.x ; p < patch->NumPixels[exposure]; p += blockDim.x) {
            int pix = patch->StartPixels[exposure] + p;

            // Get the data for this pixel
            float xp = patch.xpix[pix];
            float yp = patch.ypix[pix];
            PixFloat data = patch.data[pix];
            PixFloat ierr = patch.ierr[pix];

            // Subtracts the active and fixed Gaussians to make the residual
            PixFloat residual = ComputeResidualImage(xp, yp, data); 

            // Compute chi2 and accumulate it
            float chi2 = residual*ierr;
            chi2 *= chi2;
            accum[warp].SumChi2(chi2);
        
            // Now we loop over Active Galaxies and compute the derivatives
            for (gal = 0; gal < patch.nActiveGals; gal++) {
            float dchi2_dp[NPARAM];
            for (int j=0; j<NPARAM; j++) dchi2_dp[j]=0.0;
            for (gauss = 0; ) {    // TODO: Finish this
                ComputeGaussianDerivative(pix, residual, gal, gauss, dchi2_dp); 
            }
            accum[warp].SumDChi2dp(dchi2_dp, gal);
            }
        } // Done looping over pixels in this exposure
        __syncthreads();
    }

    // Now we're done with all exposures, but we need to sum the Accumulators
    // over all warps.
    accum[0].coadd_and_sync(accum, blockDim.x/WARPSIZE);
    accum[0].store((float *)pchi2, (float *)pdchi2_dp, nActiveGals);
    // TODO: This needs to offset the given vectors according to the Band!
    return;
}
