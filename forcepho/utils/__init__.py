# -*- coding: utf-8 -*-

import time
import json
import numpy as np

from .wcs import *
from .chain import *
from .ds9 import *
from .io import *
from .profile import *
from .catalog import *


class Logger:
    """A simple class that stores log information with similar API to logging.Logger
    """

    def __init__(self, name):
        self.name = name
        self.comments = []

    def info(self, message, timetag=None):
        if timetag is None:
            timetag = time.strftime("%y%b%d-%H.%M", time.localtime())

        self.comments.append((message, timetag))

    def serialize(self):
        log = "\n".join([c[0] for c in self.comments])
        return log


class NumpyEncoder(json.JSONEncoder):
    """
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        return json.JSONEncoder.default(self, obj)


def extract_block_diag(a, n, k=0):
    """Extract block diagonal elements from an array

    Parameters
    ----------
    a : ndarray, of shape (N, N)
        The input array

    n : int
        The size of each block

    Returns
    -------
    b : narray of shape (N//n, n, n)
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if not (n > 0):
        raise ValueError("Must have n >= 0")
    if k > 0:
        a = a[:, n*k:]
    else:
        a = a[-n*k:]

    n_blocks = min(a.shape[0]//n, a.shape[1]//n)
    new_shape = (n_blocks, n, n)
    new_strides = (n*a.strides[0] + n*a.strides[1],
                   a.strides[0], a.strides[1])

    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)


def populate_image(xpix, ypix, data):
    """Convenience utility to reshape outputs for plotting
    """

    g = (xpix >= 0) & (ypix >= 0)
    xpix = xpix[g]
    ypix = ypix[g]
    data = data[g]

    lo = np.array((xpix.min(), ypix.min())) - 0.5
    hi = np.array((xpix.max(), ypix.max())) + 0.5
    size = hi - lo
    im = np.zeros(size.astype(int)) + np.nan

    x = (xpix-lo[0]).astype(int)
    y = (ypix-lo[1]).astype(int)
    # This is the correct ordering of xpix, ypix subscripts
    im[x, y] = data
    return im, lo, hi
