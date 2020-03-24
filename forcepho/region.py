#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""region.py

Define region objects that can be used to find relevant exposures and pixels
for a given patch or area of the sky defined in celestial coordinates.
"""

import numpy as np


class Region:

    """Base class for objects that define patches on the sky
    """

    def __init__(self, *args, **kwargs):
        pass

    def contains(self, x, y, wcs, origin=0):
        """Given a set of x and y pixel coordinates, determine
        whether the pixel is within the region.

        Parameters
        ----------
        x : ndarray of shape (n_pixel, ...)
            x coordinates of the (super-)pixel
        y : ndarray of shape (n_pixel, ...)
            y coordinates of the (super-)pixel
        wcs : astropy.wcs.WCS()

        Returns
        -------
        is_in : ndarray of bool of shape (n_pixel)
        """
        return np.zeros(len(x), dtype=bool)

    @property
    def bounding_box(self):
        """A bounding box for the region, in celestial coordinates.
        """
        return None


class CircularRegion(Region):

    """An object that defines a circular region in celestial coordinates.  It
    contains methods that give a simple bounding box in celestial coordinates,
    and that can determine, given a wcs,  whether a set of pixel corners
    (in x, y) are contained within a region

    Parameters
    ----------

    ra : float
        Right Ascension of the center of the circle.  Degrees

    dec : float
        The Declination of the center of the circle.  Degrees

    radius : float
        The radius of the region, in degrees of arc.
    """

    def __init__(self, ra, dec, radius):
        self.ra = ra          # degrees
        self.dec = dec        # degrees
        self.radius = radius  # degrees of arc

    def contains(self, xcorners, ycorners, wcs, origin=0):
        """
        Parameters
        ----------
        xcorners: (nsuper, nsuper, 4) ndarray
            the full pixel x coordinates of the corners of superpixels.
        ycorners : (nsuper, nsuper, 4) ndarray
            the full pixel `y` coordinates of the corners of superpixels.
        wcs: astropy.wcs.WCS object
            header of the image including wcs information for the exposure in
            which to find pixels.

        Returns
        -------
        inreg : 2-tuple of arrays of indices
             An array that can be used to select the superpixels that have at
             least one corner within the Region
        """
        # Get the center and radius in pixel coodrinates
        # This is a bit hacky (assumes square pixels)
        # could also use the wcs.pixel_scale_matrix determinant to get pixels per degree.
        xc, yc = wcs.all_world2pix(self.ra, self.dec, origin)
        xr, yr = wcs.all_world2pix(self.ra, self.dec + self.radius, origin)
        r2 = (xc - xr)**2 + (yc - yr)**2
        d2 = (xc - xcorners)**2 + (yc - ycorners)**2
        inreg = np.any(d2 < r2, axis=-1)
        return np.where(inreg)

    @property
    def bounding_box(self):
        """Return a square bounding-box in celestial coordinates.
        The box is aligned with the celestial coordinate system.

        Returns
        -------
        bbox : ndarray of shape (2, 4)
            The ra, dec pairs of 4 corners of a square region that
            circumscribes the circular region.
        """
        dra = self.radius / np.cos(np.deg2rad(self.dec))
        ddec = self.radius
        corners = [(self.ra - dra, self.dec - ddec),
                   (self.ra + dra, self.dec - ddec),
                   (self.ra + dra, self.dec + ddec),
                   (self.ra - dra, self.dec + ddec)]
        return np.array(corners).T


class RectangularRegion(Region):

    """An object that defines a rectangular region in celestial coordinates,
    aligned with the RA and DEC axes.  It contains methods that give a simple
    bounding box in celestial coordinates, and that can determine, given a wcs,
    whether a set of pixel corners (in x, y) are contained within a region

    Parameters
    ----------
    ra_min : float
        The minimum right ascension.  Degrees

    ra_max : float
        The maximum right ascencion.  Degrees.

    dec_min : float
        The minimum Declination of the center of the circle.  Degrees

    radius : float
        The radius of the region, in degrees of arc.
    """
    def __init__(self, ra_min, ra_max, dec_min, dec_max):
        self.ra_min = ra_min       # degrees
        self.dec_min = dec_min     # degrees
        self.ra_max = ra_max       # degrees
        self.dec_max = dec_max     # degrees

    def contains(self, xcorners, ycorners, wcs, origin=0):
        """
        Parameters
        ----------
        xcorners: (nsuper, nsuper, 4) ndarray
            the full pixel x coordinates of the corners of superpixels.
        ycorners : (nsuper, nsuper, 4) ndarray
            the full pixel `y` coordinates of the corners of superpixels.
        wcs: astropy.wcs.WCS object
            header of the image including wcs information for the exposure in
            which to find pixels

        Returns
        -------
        inreg : 2-tuple of arrays of indices
             An array that can be used to select the superpixels that have at
             least one corner within the Region
        """
        # Get the corners in ra, dec coodrinates
        ra, dec = wcs.all_pix2world(xcorners, ycorners, origin)
        valid = ((ra > self.ra_min) & (ra < self.ra_max) &
                 (dec > self.dec_min) & (dec < self.dec_max))
        inreg = np.any(valid, axis=-1)
        return np.where(inreg)

    @property
    def bounding_box(self):
        """Return a square bounding-box in celestial coordinates.
        The box is aligned with the celestial coordinate system.

        Returns
        -------
        bbox : ndarray of shape (2, 4)
            The ra, dec pairs of 4 corners of a square region that
            circumscribes the circular region.  Order of the second
            dimension is SW, SE, NE, NW
        """

        corners = [(self.ra_min, self.dec_min),
                   (self.ra_max, self.dec_min),
                   (self.ra_max, self.dec_max),
                   (self.ra_min, self.dec_max)]
        return np.array(corners).T
