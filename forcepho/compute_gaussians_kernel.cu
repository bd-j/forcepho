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

class Patch {
    // Exposure data[], ierr[], xpix[], ypix[], astrometry
    // List of FixedGalaxy
    // Number of SkyGalaxy
};

class SkyGalaxy {
    // On-sky galaxy parameters
};
class Proposal {
    // List of SkyGalaxy, input for this HMC likelihood
};

class ImageGaussian {
    // 6 Gaussian parameters
    // 15 Jacobian elements (Image -> Sky)
};

class Galaxy {
    // List of ImageGaussian
    // TODO: May not need this to be explicit
};



typedef float PixFloat;
define NPARAM 7;	// Number of Parameters per Galaxy, in one band
define MAXACTIVE 30;	// Max number of active Galaxies in a patch

class Accumulator {
  public:
    float chi2;
    float dchi2_dp[NPARAM*NACTIVE];

    Accumulator() {
	chi2 = 0.0;
	for (int j=0; j<NPARAM*NACTIVE; j++) dchi2_dp[j] = 0.0;
    }
    ~Accumulator() { }
    
    // Could put the Reduction code in here
};

void Kernel() {
    Patch *patch;  // We should be given this pointer
    int band = BlockNum;   // This block is doing one band
    int warp = ThreadNum/32;  // We are accumulating in warps.

    // CreateAndZeroAccumulators();
    shared Accumulator accum[BlockDim/32]();

    // Loop over Exposures
    for (e=0; e<patch->NumExposures[band]; e++) {
        int exposure = patch->StartExposures[band]+e;
        CreateImageGaussians(exposure);

	for (p=ThreadNum; p<patch->NumPixels[exposure]; p+=BlockDim) {
	    int pix = patch->StartPixels[exposure]+p;

	    PixFloat data, ierr, residual;
	    ComputeResidualImage(pix, data, ierr, residual);
	    // This loads data and ierr, then subtracts the active
	    // and fixed Gaussians to make the residual

	    float chi2 = residual*ierr;
	    chi2 *= chi2;
	    ReduceWarp_Add(chi2, accum[warp].chi2));
	    
	    // Now we loop over Galaxies and compute the derivatives
	    for (gal = 0; ) {
		for (gauss = 0; ) {
		    ComputeGaussianDerivative(pix, residual, gal, gauss);
		}
		ReduceWarp_Add(dchi2_dp, accum[warp].dchi2_dp);
	    }
	}
    }
}
