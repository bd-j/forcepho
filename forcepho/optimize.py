# -*- coding: utf-8 -*-

import numpy as np


def split_band(patcher, arr):
    """
        - self.band_start     [NBAND] exposure index corresponding to the start of each band
        - self.band_N         [NBAND] number of exposures in each band
        - self.exposure_start [NEXP]  pixel index corresponding to the start of each exposure
        - self.exposure_N     [NEXP]  number of pixels (including warp padding) in each exposure

    Returns
    -------
    asplit : list of ndarrays
        List of band specific ndarrays of shape (n_pix_band,)
    """
    band_N_pix = np.zeros(patcher.n_bands, dtype=np.int32)
    for i, s in enumerate(patcher.band_start[:-1]):
        band_N_pix[i] = patcher.exposure_N[s:s+1].sum()
    band_N_pix[-1] = patcher.exposure_N[s+1:].sum()
    asplit = np.split(arr, np.cumsum(band_N_pix[:-1]))
    return asplit


def design_matrix(patcher, active, fixed=None, shape_cols=[]):
    """
    Returns
    -------
    Xes : list of ndarrays
       List of design matrices, each of shape (n_active, n_pix_band),
       Giving the model flux image of a given source for total flux = 1

    fixedX : list of ndarrays
       List of flux images of the fixed sources in each band.
    """

    band_N_pix = np.zeros(patcher.n_bands, dtype=np.int32)
    for i, s in enumerate(patcher.band_start[:-1]):
        band_N_pix[i] = patcher.exposure_N[s:s+1].sum()
    band_N_pix[-1] = patcher.exposure_N[s+1:].sum()

    Xes = [np.zeros((len(active), n)) for n in band_N_pix]

    if fixed is not None:
        model, q = patcher.prepare_model(active=fixed,
                                         shapes=shape_cols)
        m = patcher.data - model.residuals(q, unpack=False)
        fixedX = np.split(m, np.cumsum(band_N_pix[:-1]))
    else:
        fixedX = None

    for i, source in enumerate(active):
        model, q = patcher.prepare_model(active=np.atleast_1d(source),
                                         shapes=shape_cols)
        m = patcher.data - model.residuals(q, unpack=False)
        msplit = np.split(m, np.cumsum(band_N_pix[:-1]))
        for j, b in enumerate(patcher.bandlist):
            Xes[j][i, :] = msplit[j] / source[b]

    return Xes, fixedX


def optimize(patcher, active):
    Xes = design_matrix(patcher, active)
    ws = split_band(patcher, patcher.ierr)
    ys = split_band(patcher, patcher.data)
    for w, X, y in zip(ws, Xes, ys):
        # TODO: check matrix math
        Xp = X * w
        yp = y * w
        ATA = np.dot(Xp, Xp.T)
        Xyp = np.dot(Xp, yp[:, None])
        flux = np.linalg.solve(ATA, Xyp)



if __name__ == "__main__":
    pass