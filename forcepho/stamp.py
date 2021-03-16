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
        """Convert a sky coordinate to a pixel coordinate

        Parameters
        ----------
        sky : ndarray of floats of shape(2,)
            The celestial coordinates (ra, dec) to convert, degrees

        Returns
        -------
        pix : ndarray of floats of shape (2,)
            The pixel coordinates (x, y) of the supplied celestial coordinates
        """
        pix = np.dot(self.dpix_dsky, sky - self.crval) + self.crpix
        return pix

    def pix_to_sky(self, pix):
        """Convert a pixel coordinate to a sky coordinate

        Parameters
        ----------
        pix: ndarray of floats of shape(2,)
            The pixel coordinates (x, y) to convert, zero-indexed

        Returns
        -------
        sky : ndarray of floats of shape (2,)
            The celestial coordinates (ra, dec) of the supplied pixel
            coordinates
        """
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
        Source.render(), uses very slow methods.

        Parameters
        ----------
        source : A sources.Source instance

        compute_deriv : bool (optional, default: True)
            If True, return the gradients of the image with respect to the
            relevant free parameters for the source.

        Returns
        -------
        image : ndarray of shape (self.npix,)
            The source flux in all the pixels of the stamp

        gradients : ndarray of shape (nderiv, self.npix).  Optional.
            The gradients of the source flux in each pixel with respect to
            source parameters
        """
        return source.render(self, compute_deriv=compute_deriv,
                             **compute_keywords)


def scale_at_sky(sky, wcs, dpix=1.0, origin=1, make_approx=False):
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
    if wcs.has_distortion or make_approx:
        pos0_sky = np.array([ra, dec])
        pos0_pix = wcs.all_world2pix([pos0_sky], origin)[0]
        pos1_pix = pos0_pix + np.array([dpix, 0.0])
        pos2_pix = pos0_pix + np.array([0.0, dpix])
        pos1_sky = wcs.all_pix2world([pos1_pix], origin)[0]
        pos2_sky = wcs.all_pix2world([pos2_pix], origin)[0]

        # compute dpix_dsky matrix
        P = np.eye(2) * dpix
        St = np.array([pos1_sky - pos0_sky, pos2_sky - pos0_sky])
        CW_mat = np.linalg.solve(St, P).T

        # compute D matrix
        Winv = np.eye(2)
        Winv[0, 0] = np.cos(np.deg2rad(pos0_sky[-1]))**(-1)
        D_mat = 1.0 / 3600.0 * np.matmul(CW_mat, Winv)

    else:
        W = np.eye(2)
        W[0, 0] = np.cos(np.deg2rad(dec))
        D_mat = np.linalg.inv(wcs.pixel_scale_matrix * 3600.0)
        CW_mat = np.matmul(D_mat * 3600.0, W)

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
