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

#include <cstdint>

typedef float PixFloat;  // maybe put this elsewhere?

class Patch {
public:

    /* Image info */
    
    // Pixel data
    PixFloat *data;
    PixFloat *ierr;
    PixFloat *xpix;
    PixFloat *ypix;

    /* Indexing for the image data */
    // Number of bands and exposures is known from the CUDA grid size

    // These index the exposure_start and exposure_N arrays
    // bands are indexed sequentially, not by any band ID
    int16_t *band_start;
    int16_t *band_N;

    // These index the pixel arrays
    int *exposure_start;
    int *exposure_N;

    /* Source data */

    // Number of active sources
    // (GPU never knows about inactive sources)
    int nsources;

    // Astrometry: scale, rotation matrices (and derivatives)
    // See gaussmodel.py: convert_to_gaussians() for how these are used.
    // If D is a 2x2 matrix, then the index of matrix element (i,j) is:
    //      D[4*nsource*exposure + 4*source + 2*i + j]
    // (could also be an array of matrix structs if preferred)
    // The exposure indices for a band can be found from band_start and band_N
    float *D;
    float *CW;
    float *crpix;
    float *crval;
    float *G;

    // PSF Gaussians
    // One per sersic bin per source per exposure
    // Indexing is:
    //      psfgauss[exposure*n_psf_per_source[band]*nsource + n_psf_per_source[band]*source + psf]
    // The exposure indices for a band can be found from band_start and band_N
    PSFSourceGaussian *psfgauss;

    // The number of PSFSourceGaussians per source per exposure
    // This number is constant for a given band, hence this array is length nbands
    int8_t *n_psf_per_source;
};

class PSFSourceGaussian {
    /*
    Describes a single Gaussian that has already been convolved
    with a source Gaussian of a certain sersic radius bin.
    */

public:

    // Gaussian parameters
    float amp;
    float xcen,ycen;
    float Cxx, Cyy, Cxy;

    // The index of the sersic radius bin this Gaussian applies to
    int8_t sersic_radius_bin;
};
