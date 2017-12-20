import numpy as np
from .gaussmodel import convert_to_gaussians, get_gaussian_gradients, compute_gaussian


__all__ = ["WorkPlan", "make_workplans", "make_image",
           "negative_lnlike_multistamp"]

NDERIV = 7

def lnlike_multi(Theta, scene, plans, grad=True):
    """
    :param Theta:
        The global theta vector

    :param scene:
        A scene object that converts between sourceids and filters and indices in the giant Theta vector

    :param plans:
        A list of WorkPlan objects (corresponding roughly to stamps)

    Assumption: No two sources in a single stamp contribute to the same Theta parameter
    """
    scene.set_all_source_params(Theta)
    lnp = 0.0
    lnp_grad = np.zeros_like(Theta)
    
    for k, plan in enumerate(plans):
        plan, theta_inds, grad_inds = plan_sources(plan, scene)
        lnp_stamp, lnp_stamp_grad = plan.lnlike()
        lnp += lnp_stamp
        # TODO: test that flat[] does the right thing here
        lnp_grad[theta_inds] += lnp_stamp_grad.flat[grad_inds]

    if grad:
        return lnp, lnp_grad
    else:
        return lnp


def make_image(scene, stamp, Theta=None, use_sources=slice(None)):
    """This only works with WorkPlan object, not FastWorkPlan
    """
    if Theta is not None:
        scene.set_all_source_params(Theta)
    plan = WorkPlan(stamp)
    plan, theta_inds, grad_inds = plan_sources(plan, scene)
    plan.reset()
    plan.process_pixels()
    im = -plan.residual[use_sources].sum(axis=0).reshape(plan.stamp.nx, plan.stamp.ny)
    grad = plan.gradients[use_sources].sum(axis=0)[theta_inds, :]

    return im, grad


def plan_sources(plan, scene):
    """Add a set of sources in a scene to a work plan as active and fixed gigs
    """
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
        gig = convert_to_gaussians(source, plan.stamp)
        if coverage == 1:
            # FIXED
            fixed.append(gig)
            continue
        if coverage > 1:
            # ACTIVE
            gig = get_gaussian_gradients(source, plan.stamp, gig)
            active.append(gig)
            # get indices of parameters of the source in the giant Theta array
            theta_inds += scene.param_indices(j, plan.stamp.filtername)
            # get one-dimensional indices into the (nsource, nderiv) array of lnlike gradients
            # the somewhat crazy syntax here is because use_gradients is a Slice object
            grad_inds += ((i * NDERIV + np.arange(NDERIV))[source.use_gradients]).tolist()
            i += 1

    assert len(np.unique(theta_inds)) == len(theta_inds)
    plan.active = active
    plan.fixed = fixed
    return plan, theta_inds, grad_inds


class WorkPlan(object):

    """This is a stand-in for a C++ WorkPlan.  It takes a PostageStamp and
    lists of active and fixed GaussianImageGalaxies.  It could probably be
    formulated as a wrapper on a stamp.
    """
    
    # options for gaussmodel.compute_gaussians
    compute_keywords = {}
    nparam = 7 # number of parameters per source

    def __init__(self, stamp, active=[], fixed=[]):
        self.stamp = stamp
        self.active = active
        self.fixed = fixed
        self.reset()

    def reset(self):
        self.residual = np.zeros([self.nactive + self.nfixed, self.stamp.npix])
        self.gradients = np.zeros([self.nactive, self.nparam, self.stamp.npix])
        
    def process_pixels(self, blockID=None, threadID=None):
        """Here we are doing all pixels at once instead of one superpixel at a
        time (like on a GPU)
        """
        for i, gig in enumerate(self.active):
            for j, g in enumerate(gig.gaussians.flat):
                # get the image counts and gradients for each Gaussian in a GaussianGalaxy
                I, dI_dphi = compute_gaussian(g, self.stamp.xpix.flat, self.stamp.ypix.flat,
                                              **self.compute_keywords)
                # Store the residual.  In reality we will want to sum over
                # sources here (and divide by error) to compute chi directly
                # and avoid huge storage.
                self.residual[i, ...] -= I
                # Accumulate the *image* derivatives w.r.t. theta from each gaussian
                # This matrix multiply can be optimized (many zeros in g.derivs)
                # In reality we will want to multiply by chi and sum over pixels *HERE* to avoid storage
                self.gradients[i, :, :] += np.matmul(g.derivs, dI_dphi)

    def lnlike(self, active=None, fixed=None):
        """Returns a ch^2 value and a chi^2 gradient array of shape (nsource, nparams)
        """
        if active is not None:
            self.active = active
        if fixed is not None:
            self.fixed = fixed
        self.reset()
        self.process_pixels()
        # Do all the sums over pixels (and sources) here.  This is super inefficient.
        chi = (self.stamp.pixel_values.flatten() + self.residual.sum(axis=0)) * self.ierr

        return -0.5 * np.sum(chi*chi, axis=-1), np.sum(chi * self.ierr * self.gradients, axis=-1)


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
        return self.stamp.ierr


class FastWorkPlan(WorkPlan):
    """Like WorkPlan, but we cumulate the residuals over sources once, and cumulate chisq gradients
    """

    def reset(self):
        self.residual = self.stamp.pixel_values.flatten()
        self.gradients = np.zeros([self.nactive, self.nparam])
        
    def process_pixels(self, blockID=None, threadID=None):
        """Here we are doing all pixels at once instead of one superpixel at a
        time (like on a GPU).
        """
        for i, gig in enumerate(self.active + self.fixed):
            for j, g in enumerate(gig.gaussians.flat):
                # get the image counts for each Gaussian in a GaussianGalaxy
                self.compute_keywords['compute_deriv'] = False
                self.residual -= compute_gaussian(g, self.stamp.xpix.flat, self.stamp.ypix.flat,
                                                  **self.compute_keywords)
        for i, gig in enumerate(self.active):
            for j, g in enumerate(gig.gaussians.flat):
                self.compute_keywords['compute_deriv'] = True
                # get the image gradients for each Gaussian in a GaussianGalaxy
                I, dI_dphi = compute_gaussian(g, self.stamp.xpix.flat, self.stamp.ypix.flat,
                                              **self.compute_keywords)
                # Accumulate the *chisq* derivatives w.r.t. theta from each gaussian
                # This matrix multiply can be optimized (many zeros in g.derivs)
                self.gradients[i, :] += (self.residual * self.ierr * np.matmul(g.derivs, dI_dphi)).sum(axis=-1)

    def lnlike(self, active=None, fixed=None):
        if active is not None:
            self.active = active
        if fixed is not None:
            self.fixed = fixed
        self.reset()
        self.process_pixels()
        chi = self.residual * self.ierr

        return -0.5 * np.sum(chi*chi, axis=-1), self.gradients
