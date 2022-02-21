/* common_kernel.cc

This is the shared GPU/CPU code to compute a Gaussian mixture likelihood and
derivative.

These functions can be used to
    1. convolve on-image source PixGaussians with PSFgaussians to produce ImageGaussians,
       propogating the Jacobian of the on-sky - > on-image parameter transformation
    2. Evaluate a set of ImageGaussians on the pixel data
    3. Evaluate the on-image gradients
*/

//===============================================================================

#include "header.hh"
#include "patch.cc"

// =====================  ImageGaussian class =============================

class ImageGaussian {
  public:
    // 6 Gaussian parameters
    float amp;
    float xcen;
    float ycen;
    float fxx;
    float fyy;
    float fxy;

    // 15 Jacobian elements (Image -> Sky)
    float dA_dFlux;
    float dx_dAlpha;
    float dy_dAlpha;
    float dx_dDelta;
    float dy_dDelta;
    float dA_dQ;
    float dFxx_dQ;
    float dFyy_dQ;
    float dFxy_dQ;
    float dA_dPA;
    float dFxx_dPA;
    float dFyy_dPA;
    float dFxy_dPA;
    float dA_dSersic;
    float dA_drh;
};

// ======================  Code to Evaluate the Gaussians =========================

/// Compute the Model for one pixel from all galaxies; return the residual image
/// The one pixel is specified with xp, yp, data.
/// We have to enter a pointer to the whole list of ImageGaussians.

// TODO: n_gauss_total is a shared scalar in the calling function, but not here.
// Can we avoid the thread-based storage?  Max's advice is probably not.

CUDA_MEMBER PixFloat ComputeResidualImage(float xp, float yp,
                 PixFloat data, ImageGaussian *g, int n_gauss_total)
{

    PixFloat residual = data;

    //loop over all image gaussians g for all galaxies.
    for (int i = 0; i < n_gauss_total; i++, g++){
        // ImageGaussian *g = imageGauss+i;  // Now implicit in g++
        float dx = xp - g->xcen;
        float dy = yp - g->ycen;

        float vx = g->fxx * dx + g->fxy * dy;
        float vy = g->fyy * dy + g->fxy * dx;
        float exparg = dx*vx + dy*vy;

        if (exparg>(float)MAX_EXP_ARG) continue;
        float Gp = expf(-0.5f * exparg);

        // Here are the second-order corrections to the pixel integral
        float H = 1.0f + (vx*vx + vy*vy - g->fxx - g->fyy) / 24.f;
        float C = g->amp * Gp * H; //count in this pixel.

        residual -= C;
    }
    return residual;
}

// ====================  Code to Compute the Derivatives =====================

/// Compute the likelihood derivative for one pixel and one galaxy.
/// The one pixel is specified with xp, yp, and the residual*ierr^2.
/// We have to enter a pointer to the Gaussians for this one galaxy.
/// And we supply a pointer where we accumulate the derivatives.

// TODO: n_gal_gauss is a shared scalar in the calling function, but not here.
// Can we avoid the thread-based storage?  Max's advice is probably not.

CUDA_MEMBER void ComputeGaussianDerivative(float xp, float yp, float residual_ierr2,
                                 ImageGaussian *g, float * dchi2_dp, int n_gal_gauss)
{
    // Loop over all gaussians in this galaxy.
    for (int gauss = 0; gauss<n_gal_gauss; gauss++, g++) {
        // ImageGaussian *g = gaussian+gauss;  // Now implicit in g++

        float dx = xp - g->xcen;
        float dy = yp - g->ycen;
        float vx = g->fxx * dx + g->fxy * dy;
        float vy = g->fyy * dy + g->fxy * dx;

        float exparg = dx*vx + dy*vy;
        if (exparg>(float)MAX_EXP_ARG) continue;

        float Gp = expf(-0.5f * exparg);
        float H = 1.0f + (vx*vx + vy*vy - g->fxx - g->fyy) *(1.0f/24.0f);

        // Old code: this had divisions
        // float C = residual_ierr2 * g->amp * Gp * H;
        // float dC_dA   = C / g->amp;
        // float c_h = C / H;

        float dC_dA = -2.f*residual_ierr2 * Gp;
        float c_h = dC_dA * g->amp;
        dC_dA *= H;
        float C   = c_h * H;
        float dC_dx   = C*vx;
        float dC_dy   = C*vy;
        float dC_dfx  = -0.5f*C*dx*dx;
        float dC_dfy  = -0.5f*C*dy*dy;
        float dC_dfxy = -1.0f*C*dx*dy;

        dC_dx    -= c_h * (g->fxx*vx + g->fxy*vy) * (1.0f/12.0f);
        dC_dy    -= c_h * (g->fyy*vy + g->fxy*vx) * (1.0f/12.0f);
        dC_dfx   -= c_h * (1.0f - 2.0f*dx*vx) * (1.0f/24.0f);
        dC_dfy   -= c_h * (1.0f - 2.0f*dy*vy) * (1.0f/24.0f);
        dC_dfxy  += c_h * (dy*vx + dx*vy) * (1.0f/12.0f);

        //Multiply by Jacobian and add to dchi2_dp
        dchi2_dp[0] += g->dA_dFlux * dC_dA ;
        dchi2_dp[1] += g->dx_dAlpha * dC_dx + g->dy_dAlpha * dC_dy;
        dchi2_dp[2] += g->dx_dDelta * dC_dx + g->dy_dDelta * dC_dy;
        dchi2_dp[3] += g->dA_dQ  * dC_dA + dC_dfx * g->dFxx_dQ + dC_dfxy * g->dFxy_dQ + dC_dfy * g->dFyy_dQ;
        dchi2_dp[4] += g->dA_dPA * dC_dA + dC_dfx * g->dFxx_dPA + dC_dfxy * g->dFxy_dPA + dC_dfy * g->dFyy_dPA;
        dchi2_dp[5] += g->dA_dSersic * dC_dA;
        dchi2_dp[6] += g->dA_drh * dC_dA;
    }
}


// =================== Code to prepare the Gaussians =======================

/// This holds the information on a single source Gaussian, along with
/// info needed to compute derivatives post-convolution.

typedef struct {
    // 6 Gaussian parameters for sersic profile
    float amp;
    float xcen;
    float ycen;
    float covar; //diagonal element of sersic profile covariance. covariance = covar * I.
    matrix22 scovar_im;

    //some distortions and astrometry specific to ??source??.
    float flux;
    float G;
    float da_dn;
    float da_dr;
    matrix22 CW;
    matrix22 T;
    matrix22 dT_dq;
    matrix22 dT_dpa;
} PixGaussian;

/// This function takes a source Gaussian and a PSF Gaussian and
/// performs the convolution, creating the ImageGaussian that contains
/// the on-image gaussian plus the Jacobian to convert derivatives into
/// the on-sky parameters.

CUDA_MEMBER void GetGaussianAndJacobian(PixGaussian & sersicgauss, PSFSourceGaussian & psfgauss,
                                        ImageGaussian & gauss){

    // sersicgauss.covar is the square of the radius; it's a scalar
    sersicgauss.scovar_im = sersicgauss.covar * AAt(sersicgauss.T);

    // convolve Sersic Gaussian with PSF Gaussian
    // ----------
    matrix22 covar = sersicgauss.scovar_im + matrix22(psfgauss.Cxx, psfgauss.Cxy, psfgauss.Cxy, psfgauss.Cyy);
    matrix22 f = covar.inv();
    float detF = f.det();

    gauss.fxx = f.v11;
    gauss.fxy = f.v21;
    gauss.fyy = f.v22;

    gauss.xcen = sersicgauss.xcen + psfgauss.xcen;
    gauss.ycen = sersicgauss.ycen + psfgauss.ycen;

    float tmp = sersicgauss.G * psfgauss.amp * sqrtf(detF) * (1.0f/(2.0f*(float) M_PI));
    gauss.amp = tmp * sersicgauss.flux * sersicgauss.amp;
    // gauss.amp = sersicgauss.flux * sersicgauss.G * sersicgauss.amp * psfgauss.amp * sqrt(detF) * (1.0/(2.0*M_PI)) ;

    //now get derivatives of F
    // -----------------------
    matrix22 dSigma_dq  = sersicgauss.covar * symABt(sersicgauss.T, sersicgauss.dT_dq);
    // (sersicgauss.T * sersicgauss.dT_dq.T()+ sersicgauss.dT_dq*sersicgauss.T.T());
    matrix22 dSigma_dpa = sersicgauss.covar * symABt(sersicgauss.T, sersicgauss.dT_dpa);
    // (sersicgauss.T * sersicgauss.dT_dpa.T()+sersicgauss.dT_dpa*sersicgauss.T.T());

    matrix22 dF_dq      = -1.0f * ABA (f, dSigma_dq);  // F *  dSigma_dq * F
    matrix22 dF_dpa     = -1.0f * ABA (f, dSigma_dpa); // F * dSigma_dpa * F

    // Now get derivatives with respect to sky parameters
    // float ddetF_dq   = detF *  (covar * dF_dq).trace();
    // float ddetF_dpa  = detF * (covar * dF_dpa).trace();
    // gauss.dA_dQ      = gauss.amp /(2.0*detF) * ddetF_dq;
    // gauss.dA_dPA     = gauss.amp /(2.0*detF) * ddetF_dpa;
    // Old code: Why do we multiply and then divide by detF?

    gauss.dA_dQ      = 0.5f*gauss.amp * (covar * dF_dq).trace();
    gauss.dA_dPA     = 0.5f*gauss.amp * (covar * dF_dpa).trace();

    gauss.dA_dFlux      = tmp * sersicgauss.amp;
    gauss.dA_dSersic    = tmp * sersicgauss.flux * sersicgauss.da_dn;
    gauss.dA_drh        = tmp * sersicgauss.flux * sersicgauss.da_dr;

    // gauss.dA_dFlux   = gauss.amp / sersicgauss.flux;
    // gauss.dA_dSersic = gauss.amp / sersicgauss.amp * sersicgauss.da_dn;
    // gauss.dA_drh     = gauss.amp / sersicgauss.amp * sersicgauss.da_dr;
    // Old code: Some opportunity in the above to avoid some divisions.

    gauss.dx_dAlpha = sersicgauss.CW.v11;
    gauss.dy_dAlpha = sersicgauss.CW.v21;

    gauss.dx_dDelta = sersicgauss.CW.v12;
    gauss.dy_dDelta = sersicgauss.CW.v22;

    gauss.dFxx_dQ = dF_dq.v11;
    gauss.dFyy_dQ = dF_dq.v22;
    gauss.dFxy_dQ = dF_dq.v21;

    gauss.dFxx_dPA = dF_dpa.v11;
    gauss.dFyy_dPA = dF_dpa.v22;
    gauss.dFxy_dPA = dF_dpa.v21;
}




