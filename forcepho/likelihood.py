import numpy as np
from .gaussmodel import convert_to_gaussians, get_gaussian_gradients, compute_gaussian


__all__ = ["WorkPlan", "make_workplans", "make_image",
           "negative_lnlike_multistamp"]


def negative_lnlike_multistamp(Theta, scene=None, stamps=None):

    lnp = 0.0
    lnp_grad = np.zeros(len(Theta))
    plans, indices = make_workplans(Theta, scene, stamps)
    for wp, inds in zip(plans, indices):
        lnp_stamp, lnp_stamp_grad = wp.lnlike()
        lnp += lnp_stamp
        # TODO: test that flatten does the right thing here
        lnp_grad[inds] += lnp_stamp_grad[:, scene.use_gradients].flatten()

    return -lnp, -lnp_grad


def make_workplans(Theta, scene, stamps):
    """
    :param Theta:
        The global theta vector

    :param scene:
        A scene object that converts between sourceids and filters and indices in the giant Theta vector

    :param stamps:
        A list of stamp objects

    Assumption: No two sources in a single stamp contribute to the same Theta parameter
    """
    plans = []
    param_indices = []
    for k, stamp in enumerate(stamps):
        # Create the workplan
        active = []
        inds = []
        for source in scene.sources:
            sourceinds = scene.param_indices(source.id, stamp.filter)
            scene.set_source_params(Theta[sourceinds], source, stamp.filter)
            gig = convert_to_gaussians(source, stamp)
            gig = get_gaussian_gradients(source, stamp, gig)
            active.append(gig)
            inds += sourceinds
        assert len(np.unique(inds)) == len(inds)
        # TODO: don't reintialize the workplan every call
        plans.append(WorkPlan(stamp, active))
        param_indices.append(inds)
        
    return plans, param_indices


def make_image(Theta, scene, stamp, use_sources=slice(None)):
    """This only works with WorkPlan object, not FastWorkPlan
    """
    plans, indices = make_workplans(Theta, scene, [stamp])
    wp, inds = plans[0], indices[0]
    wp.process_pixels()
    im = -wp.residual[use_sources].sum(axis=0).reshape(stamp.nx, stamp.ny)
    grad = wp.gradients[use_sources].sum(axis=0)[inds, :]

    return im, grad


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
