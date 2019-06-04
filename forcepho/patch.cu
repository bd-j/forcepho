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
    
    // Pixel data -- all exposures, all bands
    // Note that the pixels are organized into warp-sized compact superpixels
    // and exposures are padded to blockDim.x.
    PixFloat *data;
    PixFloat *ierr;
    PixFloat *xpix;
    PixFloat *ypix;
    PixFloat *residual;

    /* Indexing for the image data */
    // Number of bands and exposures is known from the CUDA grid size
    // int n_bands = gridDim.x

    // These index the pixel arrays
    int *exposure_start;    // [expnum]
    int *exposure_N;        // [expnum]  

    // These index the exposure_start and exposure_N arrays
    // bands are indexed sequentially, not by any band ID
    // These are the expnum used elsewhere
    int16_t *band_start;    // [band]
    int16_t *band_N;        // [band]

    // ------------------ Source data --------------------
    // Number of active sources
    // (GPU never knows about inactive sources)
    int n_sources;

    // The number of radii we're using in our Sersic models
    int n_radii;   

    // ----------------------- Astrometry --------------------
    // Astrometry: scale, rotation matrices (and derivatives)
    // See gaussmodel.py: convert_to_gaussians() for how these are used.
    // If D is a 2x2 matrix, then the index of matrix element (i,j) is:
    //      D[4*nsource*exposure + 4*source + 2*i + j]
    // (could also be an array of matrix structs if preferred)
    // The exposure indices for a band can be found from band_start and band_N
    
    // D is pixels per arcsec, d(pixel x,y)/d(sky).
    // Here the sky is in arcseconds of displacement, which differs from CW
    // because of a cos(dec)
    float *D;       // [expnum][source][2][2]

    // The Coordinate Reference Point has a pixel location and a RA/Dec
    float *crpix;   // [expnum][2] -- Image pixel
    float *crval;   // [expnum][2] -- RA/Dec 

    // CW is d(pixel x,y)/d(RA,dec) expanded around CR point
    float *CW;      // [expnum][source][2][2]

    // G is the conversion from our sky flux scale into exposure counts
    float *G;       // [expnum]  


    // --------------  PSF Gaussians  ---------------------

    // The number of PSFSourceGaussians per source per exposure
    // This number is constant for a given band, hence this array is length nbands
    int *n_psf_per_source;  [band] // NOTE: This could have been type int8_t

    // Few per sersic bin per source per exposure
    // Indexing is:  TODO: Fix below
    //      psfgauss[exposure*n_psf_per_source[band]*nsource + n_psf_per_source[band]*source + psf]
    // The exposure indices for a band can be found from band_start and band_N
    PSFSourceGaussian *psfgauss;   [expnum][source][psfgauss_per_source]
    int *psfgauss_start;    [expnum]
    // psfgauss_N = n_psf_per_source*n_sources
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
    int sersic_radius_bin;
};
