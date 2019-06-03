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


__device__ void compute_gaussians(Proposal * Galaxy, Patch patch){
	
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	//loop over pixels this thread is responsible for, incrementing by number of threads in block. 
	while (tid < patch.npix) { //or similar. npix = number of pixels to do in this patch total. 
		float xp = patch.xpix[tid];
		float yp = patch.ypic[tid];
		
		//NAM TODO: load image data. 
		
		//loop over all image gaussians g. 
		for (int i = 0; i < Galaxy.nGauss; i ++){ //NAM TODO nGauss may be derived from Patch class properties. 
			ImageGaussian g = Galaxy[i]
			float dx = xp - g.xcen; 
			float dy = yp - g.ycen; 
			float vx = g.fxx * dx + g.fxy * dy;
			float vy = g.fyy * dy + g.fxy * dx;
			float Gp = exp(-0.5 * (dx*vx + dy*vy));
			float H = 1.0; 
			float root_det = 1.0; 			
			
			
			
			
			
			
			
			# --- Calculate counts ---
			if second_order:
			    H = 1 + (vx*vx + vy*vy - g.fxx - g.fyy) / 24.
			if use_det:
			    root_det = np.sqrt(g.fxx * g.fyy - g.fxy * g.fxy)
			C = g.amp * Gp * H * root_det
			
			
			
			
			
			
		}
		
		
		tid += blockDim.x; 
	
	}
	 
	
	
	
	
	
}

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

typedef struct {
    // 6 Gaussian parameters
	float xcen; 
	float ycen;
	float amp;
	float fxx; 
	float fyy;
	float fxy; 
    // 15 Jacobian elements (Image -> Sky)
} ImageGaussian;

typedef struct{
    // List of ImageGaussian
    // TODO: May not need this to be explicit
} Galaxy;
