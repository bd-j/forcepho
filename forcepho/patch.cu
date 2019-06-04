/* patch.cu

This is the data model for the Patch class on the GPU.  A Patch contains
all exposures for all bands in a patch.  It also contains per-source
astrometric information about how to convert from on-sky gaussians
to image gaussians.

PyCUDA will build this struct on the GPU side from data in the Python
Patch class (patch.py) and pass it as a pointer to each likelihood kernel
call.

TODO: decide how to fill this struct.  CUDA memcpy, or constructor kernel?

*/

class Patch {
public:
    
    // Number of active sources
    // (GPU never knows about inactive sources)
    int nsource;
    
    // Exposure data
    PixFloat * data;
    PixFloat * ierr;
    PixFloat * xpix; 
    PixFloat * ypix; 

    //astrometry: scale, rotation matrices
    // all 5 metadata arrays from convert_to_gaussians, for each exposure*source
    float 

    // PSFs
    PSFSourceGaussian *psfgauss;  // nsource*nexp*n_psf_per_source

    // Indexing array that gives start and N PSFSourceGaussians for each sersic bin for each source for each exposure
};

class PSFSourceGaussian {
public:
    float amp;
    float xcen,ycen;
    float Cxx, Cyy, Cxy;
    int sersic_radius_bin;
};
