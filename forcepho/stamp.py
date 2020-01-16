#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""stamp.py

Classes and methods for dealing with image data as python postage stamp objects
"""

import numpy as np
# from astropy import wcs

__all__ = ["PostageStamp",
           "scale_at_sky"]


class PostageStamp(object):
    """A list of pixel values and locations, along with the PSF, scale matrix,
    and astrometry

      * The PSF is an instance of PointSpreadFunction()
      * The `dpix_dsky` matrix D is defined such that :math:`p = D\, (c - c_0)
        + p_0` where :math:`p` is the pixel position, :math:`c` are the
        celestial coordinates, and :math:`c_0, p_0` indicate the CRVAL and
        CRPIX values.
      * The `scale` matrix gives pixels per arcsec
    """

    id = 1

    # --- Required Attributes that must be set directly
    # The scale matrix D (pixels per arcsecond)
    scale = np.eye(2)
    # The matrix [dpix/dRA, dpix/dDec]
    dpix_dsky = np.eye(2)
    # The sky coordinates of the reference pixel
    crval = np.zeros([2])
    # The pixel coordinates of the reference pixel
    crpix = np.zeros([2])
    # photometric conversion, physical to counts
    photocounts = 1.0
    # The point spread function
    # psf = PointSpreadFunction()

    # --- Required Attributes set by __init__ ---
    # Size of the stamp
    nx = 100
    ny = 100
    # The band name
    filtername = "dummy"

    def __init__(self, nx=100, ny=100, filtername="dummy"):
        self.nx = nx
        self.ny = ny
        self.filtername = filtername

        # The pixel values and residuals
        self.pixel_values = np.zeros([nx, ny])
        self.residuals = np.zeros([nx * ny])
        self.ierr = np.zeros_like(self.residuals)

        # Note inversion of meshgrid order
        self.ypix, self.xpix = np.meshgrid(np.arange(ny), np.arange(nx))

    def sky_to_pix(self, sky):
        pix = np.dot(self.dpix_dsky, sky - self.crval) + self.crpix
        return pix

    def pix_to_sky(self, pix):
        sky = np.dot(np.linalg.inv(self.dpix_dsky), pix - self.crpix)
        return sky + self.crval

    def coverage(self, source):
        """Placeholder method for determining whether a source qualifies as a
        fittable source in the stamp (>1), a fixed source (1), or need not be
        considered (<=0).
        """
        return 2

    @property
    def npix(self):
        return self.nx * self.ny

    @property
    def shape(self):
        return np.array([self.nx, self.ny])

    def render(self, source, compute_deriv=True, **compute_keywords):
        """Render a source on this PostageStamp.  Thin wrapper on
        Source.render()

        :param source:
            A Source object

        :param compute_deriv: (optional, default: True)
            If True, return the gradients of the image with respect to the
            relevant free parameters for the source.
        """
        return source.render(self, compute_deriv=compute_deriv,
                             **compute_keywords)


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
        self.cosdelt0 = np.cos(np.deg2rad(self.crval[-1]))
        self.W = np.array([[self.cosdelt0, 0],
                           [0., 1.]])

    def sky_to_intermediate(self, sky):
        """This does cos(dec) correction before anything else, using the dec of
        CRVAL.  Is this right enough?
        """
        return np.matmul(self.W, sky - self.crval)

    def intermediate_to_pix(self, intermediate):
        return np.matmul(self.CDinv, intermediate)

    def sky_to_pix(self, sky):
        pix = self.intermediate_to_pix(self.sky_to_intermediate(sky))
        return pix + self.crpix

    def pix_to_intermediate(self, pix):
        return np.matmul(self.CD, pix - self.crpix)

    def intermediate_to_sky(self, intermediate):
        Winv = np.linalg.inv(self.W)
        return np.matmul(Winv, intermediate) + self.crval

    def pix_to_sky(self, pix):
        return self.intermediate_to_sky(self.pix_to_intermediate(pix))

    def grad_sky_to_pix(self, sky):
        return np.matmul(self.CDinv, self.W)


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
            The 'intermediate' coordinates corresponding to the celestial
            coordinates `sky`.
        """
        alpha, delta = np.deg2rad(sky)
        ell = [np.cos(delta) * np.cos(alpha),
               np.cos(delta) * np.sin(alpha),
               np.sin(delta)]
        m = np.matmul(self.sphere_rot, np.array(ell).T)

        K = 180. / np.pi
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
        num = (np.sin(theta) * np.cos(delta_p) -
               np.cos(theta) * np.sin(delta_p) * np.cos(phi - phi_p))
        denom = -np.cos(theta) * np.sin(phi - phi_p)
        alpha = alpha_p + np.arctan2(num, denom)
        delta = np.arcsin(np.sin(theta) * np.sin(delta_p) +
                          np.cos(theta) * np.cos(delta_p) * np.cos(phi - phi_p))

        m = [np.cos(theta) * np.cos(phi),
             np.cos(theta) * np.sin(phi), np.sin(theta)]
        ell = np.matmul(self.sphere_rot.T, np.array(m))
        delta = np.arcsin(ell[-1])
        cosalpha = ell[0] / np.cos(delta)
        alpha = np.arccos(cosalpha)
        return np.rad2deg(alpha), np.rad2deg(delta)

    def pix_to_sky(self, pix):
        return self.intermediate_to_sky(self.pix_to_intermediate(pix))

    def sky_to_intermediate_gradient(self, sky):
        """Get the Jacobian matrix describing the gradients
        :math:`\partial x/\partial sky`.  
        This can be used to define a locally linear transformation matrix.
        """
        alpha, delta = np.deg2rad(sky)
        ell = [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha),
               np.sin(delta)]
        m = np.matmul(self.sphere_rot, np.array(ell).T)

        # this is partial ell / partial alpha  AND partial ell / partial delta
        dell = [[-np.cos(delta) * np.sin(alpha), np.cos(delta) * np.cos(alpha), 0],
                [-np.sin(delta) * np.cos(alpha), -np.sin(delta) * np.sin(alpha), np.cos(delta)]
                ]
        # 3 x 2 dm_i,j = partial m_i/partial c_j
        dm = np.matmul(self.sphere_rot, np.array(dell).T)

        # Could this be a matrix operation?
        K = 180. / np.pi
        x = K * m[1] / m[2]  # K * cos(phi) / tan(theta) = R_theta * cos(phi)
        y = -K * m[0] / m[2]  # -K * sin(phi) / tan(theta)
        dx_dalpha = K / m[2] * dm[1, 0] - x / m[2] * dm[2, 0]
        dx_ddelta = K / m[2] * dm[1, 1] - x / m[2] * dm[2, 1]
        dy_dalpha = -K / m[2] * dm[0, 0] - y / m[2] * dm[2, 0]
        dy_ddelta = -K / m[2] * dm[0, 1] - y / m[2] * dm[2, 1]

        W = [[dx_dalpha, dx_ddelta],
             [dy_dalpha, dy_ddelta]]

        return np.array(W)


def scale_at_sky(sky, wcs, dpix=1.0, origin=1):
    """Get the local linear approximation of the scale and CW matrix at the
    celestial position given by `sky`.  This is a simple numerical calculation

    Parameters
    ---------
    sky : iterable, length 2
        The RA and Dec coordinates in units of degrees at which to compute the
        linear approximation

    wcs : astropy.wcs.WCS() instance
        The wcs to which you want a local linear approximation

    dpix : optional, float, default; 1.0
        The number of pixels to offset to compute the local linear approx

    origin : optiona, default; 1
        The astropy wcs `origin` keyword

    Returns
    --------
    CW_mat : ndarray of shape (2, 2)
        The matrix such that (dx, dy) = CW_mat \dot (dra, ddec) where dx, dy
        are expressed in pixels and dra, ddec are exressed in degrees

    D_mat : ndarray of shape (2, 2)
        The matrix giving pixels per second of arc in RA and DEC.  Equivalent
        to the matrix inverse of 3600 times wcs.pixel_scale_matrix()
    """
    ra, dec = sky
    # get dsky for step dx, dy = dpix
    pos0_sky = np.array([ra, dec])
    pos0_pix = wcs.all_world2pix([pos0_sky], origin)[0]
    pos1_pix = pos0_pix + np.array([dpix, 0.0])
    pos2_pix = pos0_pix + np.array([0.0, dpix])
    pos1_sky = wcs.all_pix2world([pos1_pix], origin)[0]
    pos2_sky = wcs.all_pix2world([pos2_pix], origin)[0]

    # compute dpix_dsky matrix
    P = np.eye(2) * dpix
    St = np.array([pos1_sky - pos0_sky, pos2_sky - pos0_sky])
    CW_mat = (np.dot(np.linalg.inv(St), P)).transpose()

    # compute D matrix
    Winv = np.eye(2)
    Winv[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))**(-1)
    D_mat = 1.0 / 3600.0 * np.matmul(CW_mat, Winv)

    return CW_mat, D_mat


def extract_stamp(imname, errname, center=(None, None), size=(None, None),
                  center_type='pixels'):
    """Make a postage stamp around the given position using the given image
    name. This is more of an example than anything else.
    """
    from astropy.io import fits
    from astropy import wcs as apy_wcs

    im = fits.getdata(imname)
    hdr = fits.getheader(imname)
    err = fits.getdata(errname)
    crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
    crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                   [hdr['CD2_1'], hdr['CD2_2']]])

    # ---- Extract subarray -----
    center = np.array(center)
    # here we get the center coordinates in pixels
    # (accounting for the transpose above)
    if center_type == 'celestial':
        world = np.append(center, 0)
        #hdr.update(NAXIS=2)
        ast = apy_wcs.WCS(hdr)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2]
    # --- here is much mystery ---
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)
    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    crpix_stamp = np.floor(0.5 * size)
    crval_stamp = crpix_stamp + lo
    W = np.eye(2)
    if center_type == 'celestial':
        crval_stamp = ast.wcs_pix2world(crval_stamp.append(0.)[None, :], 0)[0, :2]
        W[0, 0] = np.cos(np.deg2rad(crval_stamp[-1]))

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds]
    stamp.ierr = 1. / err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny),
                                         np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval_stamp
    stamp.scale = np.matmul(np.linalg.inv(CD), W)
    stamp.pixcenter_in_full = center


def test_astrometry():
    from astropy import wcs as apy_wcs
    from astropy.io import fits
    imname = '/Users/bjohnson/Projects/nircam/mocks/image/star/sim_cube_F090W_487_001.slp.fits'
    hdr = fits.getheader(imname)
    wcs = SimpleWCS(hdr)
    awcs = apy_wcs.WCS(hdr)
    pix = np.array([924., 924.])

    sky = wcs.pix_to_sky(pix)
    asky = awcs.wcs_pix2world(np.append(pix, 0.)[None, :], 1)[0][:2]
    asky0 = awcs.wcs_pix2world(np.append(pix, 0.)[None, :], 0)[0][:2]

    rpix = wcs.sky_to_pix(sky)
    rapix = awcs.wcs_world2pix(np.append(sky, 0.)[None, :], 1)[0][:2]

    assert np.all(np.abs(rpix - rapix) < 0.1)

    print(pix, rpix, rapix)
    print(sky - asky)
