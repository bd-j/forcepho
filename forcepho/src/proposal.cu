/* proposal.cu

This is the data model for the Source class on the GPU.  
A proposal consists of a list of Sources.
Similarly, the response for the likelihood derivatives is a Band-length
vector of Response's.
*/

#ifndef PROPOSAL_INCLUDE
#define PROPOSAL_INCLUDE

/// The on-sky parameters for a single Sersic source.
class Source {
  public:
    // Source params
    float ra,dec;
    float q, pa;
    float nsersic, rh;  // These are not used in the code, but are included for flowthrough
    float fluxes[MAXBANDS];

    // The amplitudes of each of the gaussians in a mixture representation of a Sersic profile, depends on nsersic and rh
    float mixture_amplitudes[MAXRADII];
    // Gradient of mixture amplitudes with respect to rh
    float damplitude_drh[MAXRADII];
    // Gradient of the mixture amplitudes with respect to nsersic
    float damplitude_dnsersic[MAXRADII];
};


/// The response from the GPU likelihood call kernel for for a single band for a single proposal.  A full response will consist of a list of NBANDS of these responses.  
/// There can be many empty elements of this if MAXSOURCE > NSOURCE 
/// For HMC, need to pull out every NPARAMth element for the fluxes and then coadd the remaining shape gradients across bands
class Response {
  public:
    // This is the gradients for all sources in a patch in a band
    // It is ordered d/dflux, d/dra, d/dec, d/dq, d/dpa, d/dsersic, d/drh
    // and then repeats for each source. 
    float dchi2_dparam[MAXSOURCES * NPARAMS];
};

#endif
