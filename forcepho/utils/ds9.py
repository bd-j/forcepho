# -*- coding: utf-8 -*-

"""ds9.py - utilities for making ds9 region files from catalogs
"""

import os, glob

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = ["cat_to_regions", "annotate_regions"]


def cat_to_regions(cat, roi=None, ellipse=True, showid=True):
    """Turn a fpho style catalog into ds9 regions
    """

    from regions import EllipseSkyRegion, Regions

    if type(cat) is str:
        cat = fits.getdata(cat)
    if roi is None:
        roi = cat["roi"]

    regs = []
    for i, row in enumerate(cat):
        center_sky = SkyCoord(row["ra"], row["dec"], unit='deg', frame='fk5')

        if ellipse:
            sqrtq = row["q"]
            pa = 90.0 - np.rad2deg(row["pa"])
        else:
            sqrtq = 1
            pa = 0.0
        a = roi[i] / sqrtq
        b = sqrtq * roi[i]
        reg = EllipseSkyRegion(center=center_sky, height=b * u.arcsec,
                               width=a * u.arcsec, angle=pa * u.deg)
        regs.append(reg)

    reglist = Regions(regs)
    if showid:
        annotate_regions(reglist, ids=cat["id"])

    return reglist


def annotate_regions(regions, ids=None, colors=None):

    if colors is None:
        colors = np.array(len(regions) * ["green"])
    else:
        colors = np.array(colors)
    for i, r in enumerate(regions):
        if ids is not None:
            r.meta["text"] = f"{ids[i]}"
        r.visual["color"] = colors[i]
    return regions