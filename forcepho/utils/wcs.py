# -*- coding: utf-8 -*-

"""wcs.py - dealing with wcses
"""

import numpy as np
from astropy.wcs import WCS as AWCS
from astropy.io import fits

__all__ = ["scale_at_sky", "sky_to_pix"]


class FWCS:
    """Wrapper to provide common API for astropyWCS and gwcs objects.
    """

    def __init__(self, wcs=None):
        self.wcsobj = wcs

    @property
    def is_normal(self):
        return isinstance(self.wcsobj, AWCS)

    @property
    def has_distortion(self):
        if self.is_normal:
            hasdist = self.wcsobj.has_distortion
        else:
            hasdist = True
        return hasdist

    @property
    def pixel_scale_matrix(self):
        if self.is_normal:
            pixscale = self.wcsobj.pixel_scale_matrix
        else:
            raise TypeError("gWCS does not have well defined pixel scale")
        return pixscale

    def all_pix2world(self, *args, **kwargs):
        return self.pixel_to_world_values(*args, **kwargs)

    def all_world2pix(self, *args, **kwargs):
        return self.world_to_pixel_values(*args, **kwargs)

    def pixel_to_world_values(self, x, y, **extras):
        return self.wcsobj.pixel_to_world_values(x, y)

    def world_to_pixel_values(self, ra, dec, **extras):
        return self.wcsobj.world_to_pixel_values(ra, dec)

    def from_image(self, imname, extension=1):
        try:
            import asdf
            with asdf.open(imname) as fa:
                self.wcsobj = fa.search(type="WCS").node
        except:
            hdr = fits.getheader(imname, extension)
            self.wcsobj = AWCS(hdr)

    def scale_at_sky(self, sky, **kwargs):
        """Get the local linear approximation of the scale and CW matrix at the
        celestial position given by `sky`.  This is a simple numerical calculation

        Parameters
        ---------
        sky : iterable, length 2
            The RA and Dec coordinates in units of degrees at which to compute the
            linear approximation

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
            to the matrix inverse of 3600 times wcs.pixel_scale_matrix() if there are
            no distortions.
        """
        return scale_at_sky(sky, self, **kwargs)


def scale_at_sky(sky, wcs, dpix=1.0, make_approx=False):
    """Get the local linear approximation of the scale and CW matrix at the
    celestial position given by `sky`.  This is a simple numerical calculation

    Parameters
    ---------
    sky : iterable, length 2
        The RA and Dec coordinates in units of degrees at which to compute the
        linear approximation

    dpix : optional, float, default; 1.0
        The number of pixels to offset to compute the local linear approx

    wcs : instance of WCS
        The World Coordinate System  object, must have
        :py:meth:`pixel_to_world_values` method.

    Returns
    --------
    CW_mat : ndarray of shape (2, 2)
        The matrix such that (dx, dy) = CW_mat \dot (dra, ddec) where dx, dy
        are expressed in pixels and dra, ddec are exressed in degrees

    D_mat : ndarray of shape (2, 2)
        The matrix giving pixels per second of arc in RA and DEC.  Equivalent
        to the matrix inverse of 3600 times wcs.pixel_scale_matrix() if there are
        no distortions.
    """
    ra, dec = sky
    # get dsky for step dx, dy = dpix
    if getattr(wcs, "has_distortion", True) or make_approx:
        pos0_sky = np.array([ra, dec])
        pos0_pix = np.array(wcs.world_to_pixel_values(*pos0_sky))
        pos1_pix = pos0_pix + np.array([dpix, 0.0])
        pos2_pix = pos0_pix + np.array([0.0, dpix])
        pos1_sky = np.array(wcs.pixel_to_world_values(*pos1_pix))
        pos2_sky = np.array(wcs.pixel_to_world_values(*pos2_pix))

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


def sky_to_pix(ra, dec, exp=None, ref_coords=0.):
    """
    Parameters
    ----------
    ra : float (degrees)

    dec : float (degrees)

    exp : dict-like
        Must have the keys `crpix`, `crval`, and `CW` encoding the astrometry

    ref_coords : ndarray of shape (2,)
        The reference coordinates (ra, dec) for the supplied astrometry
    """
    # honestly this should query the full WCS using
    # scale_at_sky for each ra,dec pair
    crval = exp["crval"][:]
    crpix = exp["crpix"][:]
    CW = exp["CW"][:]

    i = 0
    if len(CW) != len(ra):
        CW = CW[i]
        crval = crval[i]
        crpix = crpix[i]

    sky = np.array([ra, dec]).T - (crval + ref_coords)
    pix = np.matmul(CW, sky[:, :, None])[..., 0] + crpix

    return pix

