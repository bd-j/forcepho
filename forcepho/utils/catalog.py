# -*- coding: utf-8 -*-

import os
import numpy as np
from astropy.io import fits

__all__ = ["write_table"]


def write_table(out, cats, extnames=[], **header):

    full = fits.HDUList([fits.PrimaryHDU()] + [fits.BinTableHDU(cat) for cat in cats])
    for i, hdu in enumerate(full):
        for k, v in header.items():
            try:
                hdu.header[k] = v
            except:
                pass
    for i, ext in enumerate(extnames):
        full[i+1].header["EXTNAME"] = ext
    full.writeto(out, overwrite=True)
    return full


def rectify_jades(tablename, bands=["F200W"], pixscale=0.06):

    table = fits.getdata(tablename)
    pa = np.pi - np.deg2rad(table["PA"]) # need to convert from E of N to N of E
    sqrtq = np.sqrt(table["A"] / table["B"])
    rhalf = table["A"] * sqrtq * pixscale

    dt = sourcecat_dtype(bands=bands)
    cat = np.zeros(len(pa), dtype=dt)
    cat["ra"] = table["ra"]
    cat["dec"] = table["dec"]
    cat["pa"] = pa
    cat["q"] = sqrtq
    cat["rhalf"] = rhalf
    cat["id"] = table["ID"]
    for b in bands:
        cat[b] = table[f"{b}_CIRC2"]

    return cat