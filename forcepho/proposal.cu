/* proposal.cu

This is the data model for the Source class on the GPU.  A proposal consists of a list of Sources

*/

#include <cstdint>

class Source {
    /*
    The on-sky parameters for a single Sersic source.
    */

public:

    // Source params
    float ra,dec;
    float q, pa;
    float nsersic, rh;  // These are not used
    float fluxes[MAXBANDS];
    // The amplitudes of each of the gaussians in a mixture representation of a Sersic profile, depends on nsersic and rh
    float mixture_amplitudes[MAXRADII];
    // Gradient of mixture amplitudes with respect to rh
    float damplitude_drh[MAXRADII];
    // Gradient of the mixture amplitudes with respect to nsersic
    float damplitude_dnsersic[MAXRADII];
};


class Response {
    /*
    The response from the GPU likelihood call kernel for for a single band for a single proposal.  A full response will consist of a list of NBANDS of these responses.

    There can be many empty elements of this if MAXSOURCE > NSOURCE

    Need to pull out every 7th element for the fluxes and then coadd the gradients 
    */

public:
    // This is the gradients for all sources in a patch in a band
    // It is ordered d/dflux, d/dra, d/dec, d/dq, d/dpa, d/dsersic, d/drh
    // and then repeats for each source. 
    float dchi2_dparam[MAXSOURCE * 7];
    

}