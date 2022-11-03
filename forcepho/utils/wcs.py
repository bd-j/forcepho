# -*- coding: utf-8 -*-

"""wcs.py - dealing with wcses
"""

import numpy as np
from astropy.wcs import WCS as AWCS
from astropy.io import fits

__all__ = ["FWCS"]


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

    def all_pix2world(self, x, y, origin=0):
        if self.is_normal:
            return self.wcsobj.all_pix2world(x, y, origin)
        else:
            return self.wcsobj.forward_transform(x - origin, y - origin)

    def all_world2pix(self, ra, dec, origin=0):
        if self.is_normal:
            return self.wcsobj.all_world2pix(ra, dec, origin)
        else:
            return self.wcsobj.backward_transform(ra, dec) + origin

    def from_image(self, imname, extension=1):
        try:
            import asdf
            fa = asdf.open(imname)
            self.wcsobj = fa.search(type="WCS").node
        except:
            hdr = fits.getheader(imname, extension)
            self.wcsobj = AWCS(hdr)

    def scale_at_sky(self, sky, dpix=1.0, origin=1, make_approx=False):
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
        ra, dec = sky
        # get dsky for step dx, dy = dpix
        if self.has_distortion or make_approx:
            pos0_sky = np.array([ra, dec])
            pos0_pix = self.all_world2pix([pos0_sky], origin)[0]
            pos1_pix = pos0_pix + np.array([dpix, 0.0])
            pos2_pix = pos0_pix + np.array([0.0, dpix])
            pos1_sky = self.all_pix2world([pos1_pix], origin)[0]
            pos2_sky = self.all_pix2world([pos2_pix], origin)[0]

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
            D_mat = np.linalg.inv(self.pixel_scale_matrix * 3600.0)
            CW_mat = np.matmul(D_mat * 3600.0, W)

        return CW_mat, D_mat


def scale_at_sky(sky, wcs, dpix=1.0, origin=0, make_approx=False):
    if isinstance(wcs, FWCS):
        fwcs = wcs
    else:
        fwcs = FWCS(wcs)
    return fwcs.scale_at_sky(sky, dpix=dpix, origin=origin, make_approx=make_approx)