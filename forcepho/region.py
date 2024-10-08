# -*- coding: utf-8 -*-

"""region.py

Define region objects that can be used to find relevant exposures and pixels
for a given patch or area of the sky defined in celestial coordinates.
"""

import numpy as np
from astropy.wcs import WCS

__all__ = ["Region", "CircularRegion", "RectangularRegion",
           "polygons_overlap", "polygon_contains"]


class Region:

    """Base class for objects that define patches on the sky
    """

    def __init__(self, *args, **kwargs):
        pass

    def contains(self, x, y, wcs):
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

    def overlaps(self, hdr):
        """Check for overlap of the bounding box with a rectangular image based
        on the WCS and size in its header.
        """
        wcs = WCS(hdr)
        imsize = hdr["NAXIS1"], hdr["NAXIS2"]

        # Compute image bounding box, and check if bounding boxes overlap
        pcorners = np.array([[0, imsize[0], imsize[0], 0],
                            [0, 0, imsize[1], imsize[1]]])
        ira, idec = wcs.pixel_to_world_values(pcorners[0], pcorners[1])
        im_bb = RectangularRegion(ira.min(), ira.max(), idec.min(), idec.max())
        overlap = polygons_overlap(self, im_bb)

        return overlap


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
        self.shape = "CircularRegion"

    def contains(self, xcorners, ycorners, wcs):
        """
        Parameters
        ----------
        xcorners: (nsuper, nsuper, 4) ndarray
            the full pixel x coordinates of the corners of superpixels.
        ycorners : (nsuper, nsuper, 4) ndarray
            the full pixel `y` coordinates of the corners of superpixels.
        wcs: .utils.wcs.FWCS or astropy.wcs.WCS object
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
        xc, yc = wcs.world_to_pixel_values(self.ra, self.dec)
        xr, yr = wcs.world_to_pixel_values(self.ra, self.dec + self.radius)
        r2 = (xc - xr)**2 + (yc - yr)**2
        d2 = (xc - xcorners)**2 + (yc - ycorners)**2
        inreg = np.any(d2 < r2, axis=-1)
        return np.where(inreg)

    @property
    def bounding_box(self):
        """Return a square bounding-box in celestial coordinates that
        circumscribes the circular region. The box is aligned with the celestial
        coordinate system.

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

    dec_max : float
        The maximum Declination of the center of the circle.  Degrees
    """
    def __init__(self, ra_min, ra_max, dec_min, dec_max):
        self.ra_min = ra_min       # degrees
        self.dec_min = dec_min     # degrees
        self.ra_max = ra_max       # degrees
        self.dec_max = dec_max     # degrees
        self.shape = "RectangularRegion"

    def contains(self, xcorners, ycorners, wcs):
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
        ra, dec = wcs.pixel_to_world_values(xcorners, ycorners)
        # TODO: use polygon_contains for this.  have to be careful about shapes
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


def polygons_overlap(a, b):
    """Determine whether the bounding boxes of two regions intersect, including
    whether one is fully contained within the other.  Regions must have their
    bounding polygons be convex and in consistent order (CCW *or* CW for both)

    Parameters
    ----------
    a: Region() instance
        A region with `bounding_box()` method defined.

    b : Region() instance
        A second region with `bounding_box()` method defined.

    Returns
    -------
    overlap : bool
        Whether there is any overlap.

    >>> a = RectangularRegion(58, 59, 26, 27)
    >>> b = RectangularRegion(57, 60, 25, 28)
    >>> polygons_overlap(a, b)
    True

    >>> b = RectangularRegion(60, 62, 25, 28)
    >>> polygons_overlap(a, b)
    False

    >>> b = RectangularRegion(58.5, 62, 25, 28)
    >>> polygons_overlap(a, b)
    True
    """
    for region in a, b:
        Xs, Ys = region.bounding_box
        N = len(Xs)
        inds = np.arange(N)
        normX, normY = Ys[(inds + 1) % N] - Ys[inds], Xs[inds] - Xs[(inds + 1) % N]

        ax, ay = a.bounding_box
        proj = normX[:, None] * ax[None, :] + normY[:, None] * ay[None, :]
        minA = proj.min(axis=-1)
        maxA = proj.max(axis=-1)

        bx, by = b.bounding_box
        proj = normX[:, None] * bx[None, :] + normY[:, None] * by[None, :]
        minB = proj.min(axis=-1)
        maxB = proj.max(axis=-1)

        if np.any((maxA < minB) | (maxB < minA)):
            return False

    return True


def polygon_contains(poly, points):
    """
    Parameters
    ----------
    poly : nrarray shape (nside, 2)
        x, y coordinates of polygon vertices, in clockwise or counterclockise
        order

    points : ndarray of shape (npoint, 2)
        x, y coordinates of the points to test.

    Returns
    -------
    inside : ndarray of shape (npoints,), bool
        whether the test point is inside the polygon

    >>> poly = np.array([[0,0], [1., 0], [1,1], [0, 1]])
    >>> points = np.atleast_2d([[0.5, 0.5], [0.5, 1.1], [-0.1, 0.5]])
    >>> polygon_contains(poly, points)
    [True, False, False]

    >>> poly = np.array([[0,0], [0., 1], [1,1], [1, 0]])
    >>> polygon_contains(poly, points)
    [True, False, False]

    >>> poly = np.array([[0,0], [0, 3.0], [1,1], [1, 0]])
    >>> polygon_contains(poly, points)
    [True, True, False]
    """
    vert = np.vstack([poly, poly[:1, :]])      # nside+1, 2
    ds = np.diff(vert, axis=0)                 # nside, 2
    dX = points[:, None, :] - poly[None, :, :] # npoints, nside, 2
    k = ds[:, 0] * dX[:, :, 1] - ds[:, 1] * dX[:, :, 0] # npoints, nside
    sign = np.sign(k)                          # npoints, nside
    inside = np.all((sign.T - sign[:, 0]) == 0, axis=0)
    return inside
