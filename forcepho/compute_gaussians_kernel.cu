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
	

    dchi2_dpim[0] += residual * dC_dA; //NAM TODO ??  is this right? 
    dchi2_dpim[1] += residual * dC_dx;
    dchi2_dpim[2] += residual * dC_dy;
    dchi2_dpim[3] += residual * dC_dfx;
    dchi2_dpim[4] += residual * dC_dfy;
    dchi2_dpim[5] += residual * dC_dfxy;

    // TODO: Multiply by Jacobian and add to dchi2_dp
}

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

typedef struct {
    // 6 Gaussian parameters
	float xcen; 
	float ycen;
	float amp;
	float fxx; 
	float fyy;
	float fxy; 
	// TODO: Consider whether a dummy float will help with shared memory
} ImageGaussian;

typedef struct {
    // 15 Jacobian elements (Image -> Sky)
} ImageGaussianJacobian;

typedef struct{
	int nGauss; 
	ImageGaussian * gaussians; //List of ImageGaussians
    // TODO: May not need this to be explicit
} Galaxy;




typedef float PixFloat;
#define NPARAM 7	// Number of Parameters per Galaxy, in one band 
#define MAXACTIVE 30	// Max number of active Galaxies in a patch
#define WARPSIZE 32

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
};

__device__ void CreateImageGaussians() {
    for (int gal=0; gal<nActiveGals+nFixedGals; gal++) {
		Patch patch = ;//NAM TODO
        
	    // Do the setup of the transformations		
		//Get the transformation matrix and other conversions
		D = matrix22(patch.scale[0], patch.scale[1]); //diagonal 2x2 matrix. 
		R = rot(galaxy.pa); 
		S = scale(galaxy.q); 
		matrix22 T = D * R * S; 
		CW = matrix22(patch.dpix_dsky[0], patch.dpix_dsky[1]);
		float G = patch.photocounts; 
		
		//And its derivatives with respect to scene parameters
		matrix22 dS_dq = scale_matrix_deriv(galaxy.q);
		matrix22 dR_dpa = rotation_matrix_deriv(galaxy.pa);
		matrix22 dT_dq = D * R * dS_dq; 
		matrix22 dT_dpa = D * dR_dpa * S; 	
		
        for (int s=0; s<patch->nSersicGauss; s++) {
			//get source spline and derivatives
			scovar = matrix22(galaxy.covariances[s][0], galaxy.covariances[s][1]); //diagonal elements of this gaussian's covariance matrix for sersic index s. 
			float samp = galaxy.amplitudes[s]; 
			float da_dn = galaxy.damplitude_dsersic[s];
			float da_dr = galaxy.damplitude_drh[s] ; 
			
			//pull the correct flux from the multiband array
			float flux = patch.flux[blockId.x]; //NAM TODO is this right? 
			
			//get PSF component means and covariances in the pixel space
			if (patch. )


# get PSF component means and covariances in the pixel space
if stamp.psf.units == 'arcsec':
    pcovar = fast_matmul_matmul_2x2(D, stamp.psf.covariances, D.T)
    #pmeans = np.matmul(D, stamp.psf.means)
    # FIXME need to adjust amplitudes to still sum to one?
    pamps = stamp.psf.amplitudes
elif stamp.psf.units == 'pixels':
    pcovar = stamp.psf.covariances
    #pmeans = stamp.psf.means
    pamps = stamp.psf.amplitudes
			
	    	for (int p=0; p<nPSFGauss; p++) {
				imageGauss[gal*nGalGauss+s*nPSFGauss+p] = 
		    		ConstructImageGaussian(s,p,gal);
				if (gal<nActiveGals) {
		    		imageJacob[gal*nGalGauss+s*nPSFGauss+p] = 
		    			ConstructImageJacobian(s,p,gal);
				}
	    	}
    	}
	}
}

// Shared memory is arranged in 32 banks of 4 byte stagger

__global__ void Kernel() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
    Patch * patch;  // We should be given this pointer
    int band = blockIdx.x;   // This block is doing one band
    int warp = tid / WARPSIZE;  // We are accumulating in warps.

    // CreateAndZeroAccumulators();
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

		for (p = tid ; p < patch->NumPixels[exposure]; p += blockDim.x) {
		    int pix = patch->StartPixels[exposure] + p;

		    float xp = patch.xpix[pix];
		    float yp = patch.ypix[pix];
		    PixFloat data = patch.data[pix];
		    PixFloat ierr = patch.ierr[pix];
		    PixFloat residual = ComputeResidualImage(xp, yp, data); 
		    // This loads data and ierr, then subtracts the active
		    // and fixed Gaussians to make the residual

		    float chi2 = residual*ierr;
		    chi2 *= chi2;
		    accum[warp].SumChi2(chi2);
		    /// ReduceWarp_Add(chi2, accum[warp].chi2));
	    
		    // Now we loop over Active Galaxies and compute the derivatives
		    for (gal = 0; gal < patch.nActiveGals; gal++) {
		    		float dchi2_dp[NPARAM];
				for (int j=0; j<NPARAM; j++) dchi2_dp[j]=0.0;
				for (gauss = 0; ) {
				    ComputeGaussianDerivative(pix, residual, gal, gauss, dchi2_dp); 
				}
			
				accum[warp].SumDChi2dp(dchi2_dp, gal);

				///ReduceWarp_Add(dchi2_dp, accum[warp].dchi2_dp);
		    }
		}
	__syncthreads();
    }
}
