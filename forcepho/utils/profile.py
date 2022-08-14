# -*- coding: utf-8 -*-

"""profile.py - utilities for math on sersic profiles
"""

import numpy as np
from scipy.special import gamma, gammainc, gammaincinv


__all__ = ["frac_sersic",
           "kron_radius", "isophotal_radius"]


def frac_sersic(radius, rhalf=None, sersic=2.0):
    """For a given `rhalf` and `sersic` index, compute the fraction of flux
    falling within radius
    """
    g2n = gamma(2 * sersic)
    bn = gammaincinv(2*sersic, 0.5)  # note gammainc(a, x) is normalized by gamma(a)
    x = bn * (radius / rhalf)**(1/sersic)
    return gammainc(2*sersic, x)


def kron_radius(rhalf, sersic, rmax=None):
    k = gammaincinv(2 * sersic, 0.5)
    if rmax:
        x = k*(rmax / rhalf)**(1./sersic)
        c = gammainc(3 * sersic, x) / gammainc(2 * sersic, x)
    else:
        c = gamma(3 * sersic) / gamma(2 * sersic)
    r_kron = rhalf / k**sersic * c
    return r_kron


def isophotal_radius(iso, flux, r_half, flux_radius=None, sersic=1):
    Ie = I_eff(flux, r_half, flux_radius=flux_radius, sersic=sersic)
    k = gammaincinv(2 * sersic, 0.5)
    r_iso = (1 - np.log(iso / Ie) / k)**(sersic)

    return r_iso * r_half


def I_eff(lum, rhalf, flux_radius=None, sersic=1):
    """gamma(2n, b_n) = 1/2; b_n = (re / r0)**(1/n)
    """
    two_n = 2 * sersic
    k = gammaincinv(two_n, 0.5)
    f = gamma(two_n)
    if flux_radius is not None:
        x = k * (flux_radius / rhalf)**(1. / sersic)
        f *= gammainc(two_n, x)
    conv = k**(two_n) / (sersic * np.exp(k)) / f
    Ie = lum * conv / (2 * np.pi * rhalf**2)
    return Ie