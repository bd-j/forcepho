import numpy as np

from astropy.io import fits
#from astropy import wcs


__all__ = ["PostageStamp", "TanWCS"]


class TanWCS(object):
    """Class to make various (non-linear) transformations between coordinate systems as
    defined in FITS Paper I and Paper II for TAN (gnomic) projections.

        sky < --- > intermediate < --- > pixel

    This uses FITS header keywords and assumes a CD matrix is present.
    """
    

    def __init__(self, hdr):
        
        self.crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
        self.crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
        self.CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                            [hdr['CD2_1'], hdr['CD2_2']]])
        self.CDinv = np.linalg.inv(self.CD)

        # Set up the spherical rotation matrix
        alpha_p, delta_p = np.deg2rad(self.crval)
        phi_0, theta_0, phi_p = 0, np.pi/2, np.pi
        
        r11 = -np.cos(alpha_p) * np.sin(delta_p)
        r12 = -np.sin(alpha_p) * np.sin(delta_p)
        r13 = np.cos(delta_p)
        r21 = np.sin(alpha_p)
        r22 = -np.cos(alpha_p)
        r23 = 0.
        r31 = np.cos(alpha_p) * np.cos(delta_p)
        r32 = np.sin(alpha_p) * np.cos(alpha_p)
        r33 = np.sin(delta_p)

        # This has the property that R^{-1} = R^T
        self.sphere_rot = np.array([[r11, r12, r13],
                                    [r21, r22, r23],
                                    [r31, r32, r33]])

    def sky_to_intermediate(self, sky):
        """Get intermediate coordinates given sky coordinates.

        :returns x:
            The 'intermediate' coordinates corresponding to the celestial coordinates `sky`.
        """
        alpha, delta = np.deg2rad(sky)
        ell = [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
        m = np.matmul(self.sphere_rot, np.array(ell).T)

        K = 180./np.pi
        x =  K * m[1] / m[2]
        y = -K * m[0] / m[2]

        return np.array([x, y])

    def intermediate_to_pix(self, intermediate):
        pix = np.matmul(self.CDinv, intermediate) + self.crpix
        return pix

    def sky_to_pix(self, sky):
        pix = self.intermediate_to_pix(self.sky_to_intermediate(sky))
        return pix
        
    def pix_to_intermediate(self, pix):
        intermediate = np.matmul(self.CD, pix - self.crpix)
        return intermediate

    def intermediate_to_sky(self, intermediate):
        pass

    
    def sky_to_intermediate_gradient(self, sky):
        """Get the Jacobian matrix describing the gradients \partial x/\partial
        sky.  This can be used to define a locally linear transformation matrix.
        """
        alpha, delta = np.deg2rad(sky)
        ell = [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
        m = np.matmul(self.sphere_rot, np.array(ell).T)

        # this is partial ell / partial alpha  AND partial ell / partial delta
        dell = [[-np.cos(delta) * np.sin(alpha), np.cos(delta) * np.cos(alpha), 0],
                [-np.sin(delta) * np.cos(alpha), -np.sin(delta) * np.sin(alpha), np.cos(delta)]
                ]
        dm = np.matmul(self.sphere_rot, np.array(dell).T)  # 3 x 2 dm_i,j = partial m_i/partial c_j

        # Could this be a matrix operation?
        K = 180./np.pi
        dx_dalpha = K / m[2] * dm[1, 0] - x / m[2] * dm[2, 0]
        dx_ddelta = K / m[2] * dm[1, 1] - x / m[2] * dm[2, 1]
        dy_dalpha = -K / m[2] * dm[0, 0] - y / m[2] * dm[2, 0]
        dy_ddelta = -K / m[2] * dm[0, 1] - y / m[2] * dm[2, 1]

        W = [[dx_dalpha, dx_ddelta],
             [dy_dalpha, dy_ddelta]]

        return np.array(W)
    

class PostageStamp(object):
    """A list of pixel values and locations, along with the distortion matrix,
    astrometry, and PSF.


     """

    id = 1

    # Size of the stamp
    nx = 100
    ny = 100
    npix = 100 * 100

    # The distortion matrix D
    distortion = np.eye(2)
    # The sky coordinates of the reference pixel
    crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    crpix = np.zeros([2])

    # The point spread function
    #psf = PointSpreadFunction()

    # The pixel values and residuals
    pixel_value = np.zeros([nx, ny])
    residuals = np.zeros([nx * ny])
    ierr = np.zeros_like(residuals)

    def sky_to_pix(self, sky):
        pix = np.dot(self.distortion, sky - self.crval) + self.crpix
        return pix

    def pix_to_sky(self, pix):
        pass
