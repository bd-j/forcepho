#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""likelihood.py

This module contains method for computing likelihoods of stamps given a scene.
This is all done within python using slow loops over stamps and sources
"""

import numpy as np
from .gaussmodel import convert_to_gaussians, get_gaussian_gradients, compute_gaussian


__all__ = ["lnlike_multi", "negative_lnlike_multi",
           "make_image",
           "plan_sources",
           "WorkPlan", "FastWorkPlan"
           ]

# Number of derivatives (on-sky params)
NDERIV = 7


def lnlike_multi(Theta, scene, plans, grad=True, source_meta=False):
    """Calculate the likelihood of the `plans` given the `scene` with
    parameters `Theta` This propagates the `Theta` vector into the scene, and
    then loops over the plans accumulating the likelhood (and gradients
    thereof) of each `plan`
    Assumption: No two scene sources in a single stamp/plan contribute to the
    same Theta parameter

    :param Theta:
        The global parameter vector, describing the parameters of all sources
        in the scene.

    :param scene:
        A scene object that contains a list of all sources and that converts
        between sourceids and filters and indices in the giant Theta vector

    :param plans:
        A list of WorkPlan objects, containing the data and (corresponding
        roughly to PostageStamps)

    :param grad: (optional, default: `True`)
        Switch to control whether likelihood gradients are returned

    :returns lnp:
        The total ln-likelihood of the plans/stamps given the `scene` and
        `Theta`.  Scalar

    :returns lnp_grad:
        If `grad==True`, the gradients of the ln-likelihood with respect to the
        parameters `Theta`.  ndarray of same length a `Theta`
    """
    scene.set_all_source_params(Theta)
    lnp = 0.0
    lnp_grad = np.zeros_like(Theta)

    for k, plan in enumerate(plans):
        if source_meta:
            ind = k
        else:
            ind = None
        plan, theta_inds, grad_inds = plan_sources(plan, scene, stamp_index=ind)
        lnp_stamp, lnp_stamp_grad = plan.lnlike()
        lnp += lnp_stamp
        # TODO: test that flat[] does the right thing here
        lnp_grad[theta_inds] += lnp_stamp_grad.flat[grad_inds]

    if grad:
        return lnp, lnp_grad
    else:
        return lnp


def negative_lnlike_multi(Theta, scene=None, plans=None, grad=True):
    """This wrapper that just returns the negative of lnlike_multi
    """
    lnp, lnp_grad = lnlike_multi(Theta, scene=scene, plans=plans)
    if grad:
        return -lnp, -lnp_grad
    else:
        return -lnp


def make_image(scene, stamp, Theta=None, use_sources=slice(None),
               compute_kwargs={}, stamp_index=None):
    """This only works with WorkPlan object, not FastWorkPlan

    Returns
    -------
    im : ndarray of shape (stamp.nx, stamp.ny)
        The model image

    grad : ndarray of shape (nparams, stamp.npix)
        The model image gradients
    """
    if Theta is not None:
        scene.set_all_source_params(Theta)
    plan = WorkPlan(stamp)
    plan.compute_keywords = compute_kwargs
    plan, theta_inds, grad_inds = plan_sources(plan, scene, stamp_index=stamp_index)
    plan.reset()
    plan.process_pixels()
    im = -plan.residual[use_sources].sum(axis=0).reshape(plan.stamp.nx, plan.stamp.ny)
    grad = plan.gradients[use_sources].reshape(-1, stamp.npix)[grad_inds, :]

    return im, grad


def plan_sources(plan, scene, stamp_index=None):
    """Add the sources in a scene to a work plan as lists of active and fixed
    `gaussmodel.GaussianImageGalaxies`.  Additionally returns useful arrays of
    indices for accumulating likelihood gradients in the correct elements of
    the whole lnp_grad array.  These helper arrays have the following property:

    `dlnp_dTheta[theta_inds] = (WorkPlan.lnlike()[1]).flat[grad_inds]`

    :param plan:
        A `likelihood.WorkPlan` object containig data and methods for
        calculating the ln-likelihood and its gradients.

    :param scene:
       A `sources.Scene` object containing a list of `source.Source` objects,
       as well as methods for finding the indices in the `Theta` vector
       corresponding to each `Source` object.

    :returns plan:
       The input plan with the appropriate lists of
       `gaussmodel.GaussianImageGalaxies` added as the `active` and `fixed`
       attributes.

    :returns theta_inds:
        The indices in the `Theta` array of the relevant scene parameters

    :returns grad_inds:
        The indices in the flattened `(nactive, NDERIV)` array of gradients
        returend by `WorkPlan.lnlike()` of the relevant scene parameters.
    """
    if stamp_index is None:
        stamp = plan.stamp
    else:
        stamp = stamp_index

    active, fixed = [], []
    theta_inds, grad_inds = [], []
    # Make list of all sources in the plan, keeping track of where they are
    # in the giant Theta array and where they are the (nsource, nderiv)
    # array of gradients
    i = 0
    for j, source in enumerate(scene.sources):
        coverage = plan.stamp.coverage(source)
        if coverage <= 0:
            # OFF IMAGE
            continue
        gig = convert_to_gaussians(source, stamp)
        if coverage == 1:
            # FIXED
            fixed.append(gig)
            continue
        if coverage > 1:
            # ACTIVE
            gig = get_gaussian_gradients(source, stamp, gig)
            active.append(gig)
            # get indices of parameters of the source in the giant Theta array
            theta_inds += scene.param_indices(j, plan.stamp.filtername)
            # get one-dimensional indices into the (nsource, nderiv) array of
            # lnlike gradients. The somewhat crazy syntax here is because
            # use_gradients is a Slice object
            grad_inds += ((i * NDERIV + np.arange(NDERIV))[source.use_gradients]).tolist()
            i += 1

    assert len(np.unique(theta_inds)) == len(theta_inds)
    plan.active = active
    plan.fixed = fixed
    return plan, theta_inds, grad_inds


class WorkPlan(object):

    """This is a stand-in for a C++ WorkPlan.  It takes a PostageStamp and
    lists of active and fixed `gaussmodel.GaussianImageGalaxies`.  It could
    probably be formulated as a wrapper on a stamp.
    """

    # options for gaussmodel.compute_gaussians
    compute_keywords = {}
    nparam = NDERIV  # number of parameters per source

    def __init__(self, stamp, active=[], fixed=[]):
        """Constructor.

        Parameters
        ----------
        stamp : A `data.PostageStamp` object.
            The stamp with pixel and meta data

        active :  A list of `gaussmodel.GaussianImageGalaxies` (optional)
            Sources whose likelihood gradients contribute the total likelihood
            gradient (i.e. active sources)

        fixed : A list of `gaussmodel.GaussianImageGalaxies` (optional)
            Sources that might contribute to the stamp but are assumed not to
            contribute to the total likelihood gradient (either because they are
            truly fixed or because only the far wings are contributing to the
            stamp).
        """
        self.stamp = stamp
        self.active = active
        self.fixed = fixed
        self.reset()
        self.pixel_values = self.stamp.pixel_values.flatten()

    def reset(self):
        """Reinitialize the `residual` and `gradient` arrays to zeros.
        """
        self.residual = np.zeros([self.nactive + self.nfixed, self.stamp.npix])
        self.gradients = np.zeros([self.nactive, self.nparam, self.stamp.npix])

    def process_pixels(self, blockID=None, threadID=None):
        """Here we are doing all pixels at once instead of one superpixel at a
        time (like on a GPU)
        """
        for i, gig in enumerate(self.active):
            for j, g in enumerate(gig.gaussians):
                # Get counts and gradients for each Gaussian in GaussianGalaxy
                I, dI_dphi = compute_gaussian(g, self.stamp.xpix.reshape(-1),
                                              self.stamp.ypix.reshape(-1),
                                              **self.compute_keywords)
                # Store the residual.  In reality we will want to sum over
                # sources here (and divide by error) to compute chi directly
                # and avoid huge storage.
                self.residual[i, ...] -= I
                # Accumulate the *image* derivatives w.r.t. theta from each Gaussian.
                # This matrix multiply can be optimized (g.derivs is sparse)
                # In reality we will want to multiply by chi and sum over
                # pixels *HERE* to avoid storage
                self.gradients[i, :, :] += np.matmul(g.derivs, dI_dphi)

    def lnlike(self, active=None, fixed=None):
        """Returns a scalar chi^2 value and a chi^2 gradient array of
        shape (nactive, NDERIV)
        """
        if active is not None:
            self.active = active
        if fixed is not None:
            self.fixed = fixed
        self.reset()
        self.process_pixels()
        # Do all the sums over pixels (and sources) here.  This is inefficient.
        chi = (self.pixel_values + self.residual.sum(axis=0)) * self.ierr

        return -0.5 * np.dot(chi, chi.T), np.dot(self.gradients, chi*self.ierr)

    def make_image(self, use_sources=slice(None)):
        self.reset()
        self.process_pixels()
        return self.residual[use_sources, ...].sum(axis=0).reshape(self.stamp.nx, self.stamp.ny)

    @property
    def nactive(self):
        return len(self.active)

    @property
    def nfixed(self):
        return len(self.fixed)

    @property
    def ierr(self):
        return self.stamp.ierr.reshape(-1)


class FastWorkPlan(WorkPlan):
    """Like WorkPlan, but we cumulate the residuals over sources once, and
    cumulate chisq gradients
    """

    def reset(self):
        self.residual = self.stamp.pixel_values.flatten()
        self.gradients = np.zeros([self.nactive, self.nparam])
        self.chi = None

    def process_pixels(self, blockID=None, threadID=None):
        """Here we are doing all pixels at once instead of one superpixel at a
        time (like on a GPU).
        """
        # Loop over active and fixed to get the residual image
        for i, gig in enumerate(self.active + self.fixed):
            for j, g in enumerate(gig.gaussians):
                # get the image counts for each Gaussian in a GaussianGalaxy
                self.compute_keywords['compute_deriv'] = False
                self.residual -= compute_gaussian(g, self.stamp.xpix.flat,
                                                  self.stamp.ypix.flat,
                                                  **self.compute_keywords)
        self.chi = self.residual * self.ierr
        self.chivar = self.chi * self.ierr
        # Loop only over active to get the chisq gradients
        for i, gig in enumerate(self.active):
            for j, g in enumerate(gig.gaussians):
                self.compute_keywords['compute_deriv'] = True
                # get the image gradients for each Gaussian in a GaussianGalaxy
                I, dI_dphi = compute_gaussian(g, self.stamp.xpix.flat,
                                              self.stamp.ypix.flat,
                                              **self.compute_keywords)
                # Accumulate the *chisq* derivatives w.r.t. theta from each Gaussian
                # This matrix multiply can be optimized (g.derivs is sparse)
                self.gradients[i, :] += np.matmul(g.derivs, np.matmul(dI_dphi, self.chivar))

    def lnlike(self, active=None, fixed=None):
        if active is not None:
            self.active = active
        if fixed is not None:
            self.fixed = fixed
        self.reset()
        self.process_pixels()

        return -0.5 * np.dot(self.chi, self.chi.T), self.gradients
