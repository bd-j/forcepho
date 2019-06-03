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


__device__ PixFloat ComputeResidualImage(float xp, float yp, PixFloat data, Patch patch, Galaxy galaxy); //NAM do we need patch, galaxy? 
{
	
	
	residual = 0.0; 
	
	//loop over all image gaussians g. 
	for (int i = 0; i < galaxy.nGauss; i ++){ //NAM TODO nGauss may be derived from Patch class properties. 
		ImageGaussian g = galaxy.Gaussians[i]
		float dx = xp - g.xcen; 
		float dy = yp - g.ycen; 
		float vx = g.fxx * dx + g.fxy * dy;
		float vy = g.fyy * dy + g.fxy * dx;
		float Gp = exp(-0.5 * (dx*vx + dy*vy));
		float H = 1.0; 
		float root_det = 1.0; 			
		
		H = 1.0 + (vx*vx + vy*vy - g.fxx - g.fyy) / 24.0; 
		float C = g.amp * Gp * H * root_det; //count in this pixel. 
		
		residual += data - C; 
	}
	return residual;
	
	
}

__device__ void ComputeGaussianDerivative(pix, residual, gal, gauss, float * dchi2_dp) //NAM why are we passing in residual? it's been accumulated over ImageGaussians earlier... we need to repeat work here to get isolated Image Gaussian
{
	float xp = patch.xpix[pix];
	float yp = patch.ypic[pix];
	
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
	

    dchi2_dp[0] += residual * dC_dA; //NAM TODO ??  is this right? 
    dchi2_dp[1] += residual * dC_dx;
    dchi2_dp[2] += residual * dC_dy;
    dchi2_dp[3] += residual * dC_dfx;
    dchi2_dp[4] += residual * dC_dfy;
    dchi2_dp[5] += residual * dC_dfxy;
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
    void SumDChi2dp(float *_dchi2_dp) { 
	for (int j=0; NPARAM*NACTIVE; j++) 
	    warpReduceSum(dchi2_dp+j, _dchi2_dp[j]); 
    }
};

void Kernel() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
    Patch * patch;  // We should be given this pointer
    int band = blockIdx.x;   // This block is doing one band
    int warp = tid / WARPSIZE;  // We are accumulating in warps.

    // CreateAndZeroAccumulators();
    __shared__ Accumulator accum[blockDim.x/WARPSIZE]();

    // Loop over Exposures
    for (e = 0; e < patch->NumExposures[band]; e++) {
        int exposure = patch->StartExposures[band] + e;
        CreateImageGaussians(patch, exposure);

		for (p = tid ; p < patch->NumPixels[exposure]; p += blockDim.x) {
		    int pix = patch->StartPixels[exposure] + p;

		    float xp = patch.xpix[pix];
		    float yp = patch.ypic[pix];
		    PixFloat data = patch.data[pix];
		    PixFloat ierr = patch.ierr[pix];
		    PixFloat residual = ComputeResidualImage(xp, yp, data); 
		    // This loads data and ierr, then subtracts the active
		    // and fixed Gaussians to make the residual

		    float chi2 = residual*ierr;
		    chi2 *= chi2;
		    accum[warp].SumChi2(chi2);
		    /// ReduceWarp_Add(chi2, accum[warp].chi2));
	    
		    // Now we loop over Galaxies and compute the derivatives
		    for (gal = 0; ) {
				for (gauss = 0; ) {
				    ComputeGaussianDerivative(pix, residual, gal, gauss, dchi2_dp); 
				}
			
				accum[warp].SumDChi2dp(dchi2_dp);

				///ReduceWarp_Add(dchi2_dp, accum[warp].dchi2_dp);
		    }
		}
    }
}
