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
    for i in range(patcher.n_bands-1):
        st, sp = patcher.band_start[i:i+2]
        band_N_pix[i] = patcher.exposure_N[st:sp].sum()
    band_N_pix[-1] = patcher.exposure_N[patcher.band_start[-1]:].sum()

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


def optimize_one(X, w, y, fixedX=0):
    Xp = X * w
    yp = (y - fixedX) * w
    ATA = np.dot(Xp, Xp.T)
    Xyp = np.dot(Xp, yp[:, None])
    flux = np.linalg.solve(ATA, Xyp)
    return flux, ATA


def optimize(patcher, active, fixed=None, shape_cols=[], return_all=True):
    """Do a simple wieghted least-squares to get the maximum likelihood fluxes,
    conditional on source parameters.

    Returns
    -------
    fluxes : list of ndarrays
        List of ndarrays of shape (n_source,) giving the maximum likelihoood
        fluxes of each source. The list has same length and order as
        `patcher.bandlist`

    precisions : list of ndarrays
        List of ndarrays of shape (n_source, n_source) giving the flux precision
        matrix in each band (i.e. the inverse of the flux covariance matrix)
    """
    fluxes, models, precisions = [], [], []
    Xes, fixedX = design_matrix(patcher, active,
                                shape_cols=shape_cols, fixed=fixed)
    ws = split_band(patcher, patcher.ierr)
    ys = split_band(patcher, patcher.data)
    fX = 0.0
    for i, (w, X, y) in enumerate(zip(ws, Xes, ys)):
        if fixedX is not None:
            fX = fixedX[i]
        flux, precision = optimize_one(X, w, y, fixedX=fX)
        fluxes.append(np.squeeze(flux))
        precisions.append(precision)
        if return_all:
            model = np.dot(flux.T, X)
            models.append(np.squeeze(model))

    if return_all:
        return fluxes, precisions, models, fixedX
    else:
        return fluxes, precisions


if __name__ == "__main__":
    pass