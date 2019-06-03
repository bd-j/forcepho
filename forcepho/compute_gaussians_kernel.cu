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
    // Exposure data, ierr, xpix, ypix, astrometry
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
