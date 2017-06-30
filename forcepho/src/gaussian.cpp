

// ============================================================================
// External to this code, the handler needs to generate the postage stamps,
// notably the list of superpixels and their inverse variances, and the list
// of GaussianGalaxies that overlap the stamps.
// The external code then needs to set up the WorkList that contain the itemized
// pairings of superpixels and GaussianGalaxies that will be computed, as well
// as the output buffers to hold the answers.

// We plan to take multiple proposal steps in a single postage stamp.  That means
// that the pixels and the evalution of the fixed Gaussians can be held fixed.
// Only the activeGaussians will be updated.

// The GaussianGalaxies are sorted into two lists, 1) those that are being varied and
// and therefore need derivatives computed, 2) those that are fixed and just need to
// be subtracted once from the images when computing the residuals.

// In the first proposal, we will compute the Fixed Gaussians and subtract them from
// pixels[] array.  In later proposals, the WorkList will have no Fixed Gaussians.

// For a GPU, we plan that each SuperPixel is a threadblock, and each thread computes
// for one pixel.  For AVX, the SuperPixel would be a small set of vectors.


#define NDERIV 15    // The number of dGaussParam/dSceneParam elements.
// We have a lot of zero elements, so we will not store the full matrix.
// Consider having preprocessor names for the enumerations.

class Gaussian {
    /// This is description of one Gaussian, in pixel locations,
    /// as well as the derivatives of this Gaussian wrt to the Scene model parameters.
    float amplitude;
    float Fxx, Fxy, Fyy;
    float xcen, ycen;
    // 
    float dGaussian_dScene[NDERIV];
}

#define NGAUSS 20  // The maximum number of Gaussians in a Galaxy

class GaussianGalaxy {
     // The statement of a Galaxy model converted into a series of Ng Gaussians,
     // in pixel space (i.e., after image distortions) and after convolution with the PSF.
     // Further, the derivatives of the 6*Ng Gaussian parameters wrt to the Galaxy parameters.
     Gaussian g[NGAUSS];
     int nGauss;    // It's possible that we won't use all NGAUSS
     int id;     // The Galaxy ID number (i.e., in the parent list of Galaxies).
};

// ============== Classes for the Work Plan ====================


class WorkList {
    // This is one item in the WorkList.  It will specify one SuperPixel 
    // and a set of Active and Fixed Galaxies.
    int superpixel;	// Offset into the pixel & ivar list
    float x0, y0;	// The x,y pixel location in stamp frame of the 0 pixel in the superpixel
    int filter;
    int galaxyID;
    int nActive, start_Active;    // Offset into list of Active GaussianGalaxies worklist
    int nFixed, start_Fixed;    // Offset into list of Fixed GaussianGalaxies worklist
};

#define NPARAM   7    // The number of derivative parameters for one Galaxy and one filter
#define NFISHER (NPARAM*(NPARAM+1)/2)    // The size of an upper triangular matrix.
#define NPFISH (NPARAM+NFISHER)

class OutputGalaxyDeriv {
     // The derivative outputs for a Galaxy+Filter
     float dchi2_dSceneParam[NPARAM];
     float Fisher_SceneParam[NFISHER];
}


class WorkPlan {
    // We unify all of the handler instructions into a Class called a 
    // WorkPlan, so that we can have multiple Plans on the CPU side and no global 
    // variables!

    int SuperPixelSize;	// The quantification of the SuperPixels
    int SuperPixelX;	// SuperPixels will be this wide in X (Y is Size/X)

    // TODO: May need to offer support for multiple sizes of superpixels, because SW!=LW!=HST.
    // But this may depend on GPU planning regarding thread blocks.
    // We may just launch the LW and SW separately?

    // Here are the pixels that are being loaded in.

    float *pixels;		// The concatenated list of all of the Postage Stamp pixels
    float *ivars;		// The concatenated list of all of the Postage Stamp ivars
    int nPixel;		// The number of pixels
    // These two lists will be organized by superpixel.

    GaussianGalaxy *active_galaxies;	// The concatenated list of all of varying GaussianGalaxies
    GaussianGalaxy *fixed_galaxies;     // The concatenated list of all of fixed GaussianGalaxies
    // The concept is that we're evaluating a postage stamp, but only some of the galaxies
    // that touch the stamp are varying.  We need to create the fixed image too.
    // Note that GaussianGalaxies are unique to an exposure -- a given Galaxy appears many times.


    // Here is the main listing of the work to be done.
    // We generate an explicit pairing of every superpixel and every GaussianGalaxy

    WorkList *WL;		// The list of the work
    int nWorkList;		// Size of this array

    int *worklist_ActiveGaussianGalaxy;   
	// The list of indices of Active GaussianGalaxies to evaluate for each SuperPixel
    int nWorkActive;		// Size of this array

    int *worklist_FixedGaussianGalaxy;   
	// The list of indices of Fixed GaussianGalaxies to evaluate for each SuperPixel
    int nWorkFixed;		// Size of this array

    // ----------------------------------------------------------------------------

    // Here are buffers to store the outputs.

    float chi2;    // TODO: need to think more about how to do this reduction.

    float *pixel_residual;	   // The residual image after everything is subtracted
    // Both of these have size nPixel

    int *worklist_out;    
    	// size nWorkActive; the element of the Output list to add the results

    OutputGalaxyDeriv *output;	// The Output for each galaxyID and filter.
    int nOutput; 	// The number of unique filter & galaxyID pairs

    // Ultimately, we return only the output array and (optionally) the pixel_residual.

    // TODO: Need constructor and destructors for all of this
};

// ============================================================================
// Now we compute!  Here's an approximation to a CUDA Kernel

void WorkPlan:ProcessPixel(int blockID, int threadID) {
    // Given an element in the worklist, do all of the work.

    // TODO: Probably some of the control stuff can be done in the shared memory on the GPU
    // We could load all of the Gaussians into shared memory, do one synchronization, and 
    // then launch the rest.  Note that concatenating the relevant Gaussians could also 
    // reduce the memory pressure on some of the loops by avoiding the double nested control.

    // On a GPU, these go in shared memory:
    float dI_dSceneParam[NPARAM][SuperPixelSize];
    float pixel_value[SuperPixelSize];
    float vector_to_reduce[SuperPixelSize];

    // Figure out what pixel we're doing.
    int pix_blockstart = WL[blockID].superpixel*SuperPixelSize 
    int pix = pix_blockstart + threadID;
    assert(pix<nPixel);

    pixel_value[threadID] = pixels[pix];
    float ierror = sqrt(ivars[pix]);

    // Figure out where it is
    // TODO: Might be cheaper ways to code this math
    float xpix = WL[blockID].x0 + (threadID%SuperPixelX)
    float ypix = WL[blockID].y0 + (threadID/SuperPixelX)

    // TODO: The following loop control might be done with some single shared variable *if* we
    // are ok to force synchronization between warps.

    // Compute all of the Fixed Gaussians
    int end_Fixed = WL[blockID].start_Fixed+WL[blockID].nFixed
    for (int n=WL[blockID].start_Fixed; n<end_Fixed; n++) {
	int ng = worklist_FixedGaussianGalaxy[n];
	for (int ig=fixed_galaxies[ng].nGauss-1; ig>=0; ig--) {
	    computeGaussian(fixed_galaxies[ng].g[ig], xpix, ypix, pixel_value);
	}
    }

    // If we had any Fixed Gaussians, overwrite the pixels array, 
    // so that we don't have to do these on subsequent Scenes.
    if (WL[blockID].nFixed>0) pixels[pix] = pixel_value[threadID];

    // Compute all of the Active Gaussians
    int end_Active = WL[blockID].start_Active+WL[blockID].nActive
    for (int n=WL[blockID].start_Active; n<end_Active; n++) {
	int ng = worklist_ActiveGaussianGalaxy[n];
	for (int ig=active_galaxies[ng].nGauss-1; ig>=0; ig--) {
	    computeGaussian(active_galaxies[ng].g[ig], xpix, ypix, pixel_value);
	}
    }

    // Now we have the residual image in pixel_value.
    pixel_residual[pix] = pixel_value[threadID];   // Save the actual residual value

    // Rescale the residual by the inverse error
    pixel_value *= ierror;

    // Compute chi2 and add to some accumulator
    vector_to_reduce[threadID] = pixel_value*pixel_value;
    SynchronizeThreads();
    reduce_vector_and_cumulate(chi2, vector_to_reduce);

    // Time to compute the derivatives for each Active Galaxy
    int end_Active = WL[blockID].start_Active+WL[blockID].nActive
    for (int n=WL[blockID].start_Active; n<end_Active; n++) {
	// Set up the accumulators for dI/dSceneParam; 
	for (int i=0; i<NPARAM; i++) dI_SceneParam[i][threadID]=0.0;

	int ng = worklist_ActiveGaussianGalaxy[n];
	for (int ig=active_galaxies[ng].nGauss-1; ig>=0; ig--) {
	    computeGaussianDeriv(active_galaxies[ng].g[ig], xpix, ypix, dI_dSceneParam, ierror);
	}
	OutputGalaxyDeriv *out = output + workplace_output[n];
	computeDeriv(out, dI_dSceneParam, pixel_value);
    }

    // Compute the dchi2/dSceneParam and the Fisher elements for this pixel.
    // Add to the global sums for this filter & galaxy
}


void WorkPlan:computeGaussian(Gaussian &g, float xpix, float ypix, float *pixel_value) {
    // For one Gaussian and one pixel, do the computation of the intensity and 
    // subtract it from the pixel value.
    dx = xpix-g.xcen;
    dy = ypix-g.ycen;
    vx = Fxx*dx + Fxy*dy;
    vy = Fyy*dy + Fxy*dx;
	// Gp = exp(-0.5*(dx*dx*Fxx + 2.0*dx*dy*Fxy + dy*dy*Fxy)); 
    	// or vx*dx+vy*dy
    Gp = -0.5*(vx*dx + vy*dy);
    C = 1 + (1.0/3.0)*(vx*vx + vy*vy - Fxx - Fyy);
    Gp = exp(Gp);    // TODO: could switch to exp2, if faster
    pixel_value -= amplitude*Gp*C;
    return;
}

void WorkPlan:computeGaussianDeriv(Gaussian &g, float xpix, float ypix, float dI_dSceneParam[], ierror) {
    // For one Gaussian and one pixel, do the computation of all derivatives.
    dx = xpix-g.xcen;
    dy = ypix-g.ycen;
    vx = Fxx*dx + Fxy*dy;
    vy = Fyy*dy + Fxy*dx;
	// Gp = exp(-0.5*(dx*dx*Fxx + 2.0*dx*dy*Fxy + dy*dy*Fxy)); 
    	// or vx*dx+vy*dy
    Gp = -0.5*(vx*dx + vy*dy);
    C = 1 + (1.0/3.0)*(vx*vx + vy*vy - Fxx - Fyy);
    Gp = exp(Gp);    // TODO: could switch to exp2, if faster
    Gp *= ierror;    // We are scaling all of the derivatives by the inverse pixel error
    dI_dA = Gp*C;
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    AGp = amplitude*Gp;
    dI_dmux = AGp*(vx*C - (2.0/24.0)*(Fxx*vx + Fxy*vy);
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    dI_dmuy = AGp*(vy*C - (2.0/24.0)*(Fyy*vy + Fxy*vx);
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    dI_dFxx = AGp*(-0.5*dx*dx*C + (1.0/24.0)*(2.0*dx*vx-1.0));
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    dI_dFyy = AGp*(-0.5*dy*dy*C + (1.0/24.0)*(2.0*dy*vy-1.0));
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    dI_dFxy = AGp*(-dx*dy*C + (2.0/24.0)*(dy*vx+dx*vy));
    // Multiply by dGaussian_dScene to get the derivatives dI_dScene.
    // TODO: Need to provide the multiplies
}

void WorkPlan:computeDeriv(TODO) {
    // Given the derivatives dI_dSceneParam for this GaussianGalaxy, compute the 
    // dchi2_dSceneParam values and the Fisher values.  Sum over pixels and add them 
    // to the correct location.
    vector_to_reduce[threadID] = dI_SceneParam[0]*dI_dSceneParam[0];
    SynchronizeThreads();
    reduce_vector_and_cumulate(out->dchi2_Fisher[0], vector_to_reduce);
    // TODO: And so on
    vector_to_reduce[threadID] = pixel_value*dI_dSceneParam[0];
    SynchronizeThreads();
    reduce_vector_and_cumulate(out->dchi2_dSceneParam[0], vector_to_reduce);
    // TODO: And so on
}

void WorkPlan:reduce_vector_and_cumulate(float &chi2, float vector_to_reduce[]) {
    // This routine takes a vector of size equal to the thread block, and 
    // reduces it to one number, then adds that number to the first value 
    // with an AtomicAdd.
}



// =========================  Random sample code from an external driver =========
// The below belongs in the handler code, not here.


class Galaxy {
    // This has the on-sky parameters that we are varying.
    float ra, dec;
    float axis_ratio, pos_angle;
    float sersic, ps_flux;
    float flux[NBAND];
};
Galaxy *galaxies;   // The list of all of the Galaxies

class ExposurePostageStamp {
    int filter;
    int nx, ny;
    int nPixel, start_Pixel;   // Offset into the pixel and ivar lists
    int nActive, start_Active;   // Offset into the GaussianGalaxy list
    int nFixed, start_Fixed;   // Offset into the GaussianGalaxy list
}

ExposurePostageStamp *exposures;

// The handler needs to generate the list of ExposurePostageStamps,
// the list of superpixels and ivars, and the list of GaussianGalaxies.
// This involves a lot of use of the distortion maps and bounding boxes to prune down
// the lists.

