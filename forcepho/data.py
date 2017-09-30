import numpy as np

#from astropy import wcs


__all__ = ["PostageStamp", "TanWCS"]


class PostageStamp(object):
    """A list of pixel values and locations, along with the scale matrix,
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


class SimpleWCS(object):
    """Class to make various transformations between coordinate systems for a
    simple TAN WCS projection.

        sky < --- > intermediate < --- > pixel

    This uses FITS header keywords and assumes a CD matrix is present.

    The fundamental thing we do here is assume that the CD matrix gives
       [[cos(delta) * dalpha / dp_x, cos(delta) * dalpha / dp_y],
        [ddelta/dp_x, ddelta/dp_x]]

    This is valid for small images and/or not far from the original CRVAL and
    CRPIX of the projection
    """

    def __init__(self, hdr):
        
        self.crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
        self.crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
        self.CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                            [hdr['CD2_1'], hdr['CD2_2']]])
        self.CDinv = np.linalg.inv(self.CD)

    def sky_to_intermediate(self, sky):
        """This does cos(dec) correction before anything else, using the dec of
        the input sky coordinate (possible that it's better to use the dec of
        crval?)
        """
        W = np.array([[np.cos(np.deg2rad(sky[-1])), 0],
                      [0., 1.]])
        return np.matmul(W, sky - self.crval)

    def intermediate_to_pix(self, intermediate):
        return np.matmul(self.CDinv, intermediate)

    def sky_to_pix(self, sky):
        return self.intermediate_to_pix(self.sky_to_intermediate)) + self.crpix

    def pix_to_intermediate(self, pix):
        return np.matmul(self.CD, pix - self.crpix)

    def intermediate_to_sky(self, intermediate):
        Winv = np.array([[1./np.cos(np.deg2rad(sky[-1])), 0],
                        [0., 1.]])
        return np.matmul(Winv, intermediate) + self.crval

    def pix_to_sky(self, pix):
        return self.intermediate_to_sky(self.pix_to_intermediate(pix))

    def grad_sky_to_pix(self, sky):
        W = np.array([[np.cos(np.deg2rad(sky[-1])), 0],
                      [0., 1.]])
        return np.matmul(seld.CDinv, W)


class TanWCS(object):
    """Class to make various transformations between coordinate systems as
    defined in FITS Paper I and Paper II for TAN (gnomic) projections.

        sky < --- > intermediate < --- > pixel

    This uses FITS header keywords and assumes a CD matrix is present.

    This code appears to be broken, possibly for numerical reasons or a typo in
    the shperical transforms...?
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
        x =  K * m[1] / m[2]  # K * cos(phi) / tan(theta) = R_theta * cos(phi)
        y = -K * m[0] / m[2]  # -K * sin(phi) / tan(theta)

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
        # Yuck.
        x, y = intermediate
        phi = np.arctan2(-x, y)
        Rtheta = np.hypot(x, y)
        theta = np.arctan(180./np.pi / Rtheta)

        # Set up the spherical rotation matrix
        alpha_p, delta_p = np.deg2rad(self.crval)
        phi_0, theta_0, phi_p = 0, np.pi/2, np.pi
        num = np.sin(theta) * np.cos(delta_p) - np.cos(theta) * np.sin(delta_p) * np.cos(phi - phi_p)
        denom = -np.cos(theta) * np.sin(phi - phi_p)
        alpha = alpha_p + np.arctan2(num, denom)
        delta = np.arcsin(np.sin(theta) * np.sin(delta_p) + np.cos(theta) * np.cos(delta_p) * np.cos(phi-phi_p))
        
        m = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        ell = np.matmul(self.sphere_rot.T, np.array(m))
        delta = np.arcsin(ell[-1])
        cosalpha = ell[0] / np.cos(delta)
        alpha = np.arccos(cosalpha)
        return np.rad2deg(alpha), np.rad2deg(delta)

    def pix_to_sky(self, pix):
        return self.intermediate_to_sky(self.pix_to_intermediate(pix))

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
        x =  K * m[1] / m[2]  # K * cos(phi) / tan(theta) = R_theta * cos(phi)
        y = -K * m[0] / m[2]  # -K * sin(phi) / tan(theta)
        dx_dalpha = K / m[2] * dm[1, 0] - x / m[2] * dm[2, 0]
        dx_ddelta = K / m[2] * dm[1, 1] - x / m[2] * dm[2, 1]
        dy_dalpha = -K / m[2] * dm[0, 0] - y / m[2] * dm[2, 0]
        dy_ddelta = -K / m[2] * dm[0, 1] - y / m[2] * dm[2, 1]

        W = [[dx_dalpha, dx_ddelta],
             [dy_dalpha, dy_ddelta]]

        return np.array(W)


def test_astrometry():
    from astropy.io import fits
    imname = '/Users/bjohnson/Projects/nircam/mocks/image/star/sim_cube_F090W_487_001.slp.fits'
    hdr = fits.getheader(imname)
    wcs = TanWCS(hdr)
    px, py = 100., 200.
    pix = array([px, py])
    sky = np.array(wcs.pix_to_sky(pix))
    print(pix, wcs.sky_to_pix(sky))

    test_sky = np.array(wcs.pix_to_sky(pix + 1.))

    # Do it with the full non-linear
    true_pix = wcs.sky_to_pix(test_sky)
    
    # Do it with the local distortion matrix
    W = wcs.sky_to_intermediate_gradient(sky)
    test_pix = np.dot(wcs.CDinv, np.dot(W, test_sky - wcs.crval)) + wcs.crpix

    # Do it with cos(dec) approx
    pW = np.eye(2)
    pW[0, 0] = np.cos(np.deg2rad(sky[-1]))
    test_pix2 = np.dot(wcs.CDinv, np.dot(pW, test_sky - wcs.crval)) + wcs.crpix
