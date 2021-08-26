# -*- coding: utf-8 -*-

"""superscene.py

Parent side classes for handling the master catalog, including checking regions
in and out.  Also methods for making prior bounds, and computing ROIs.
"""


import os, sys
import json

import numpy as np

from scipy.special import gamma, gammainc, gammaincinv
from scipy.spatial import cKDTree
from scipy.linalg import block_diag
from astropy.io import fits
from astropy.coordinates import SkyCoord

from .region import CircularRegion
from .sources import Galaxy
from .utils import rectify_catalog, read_config


__all__ = ["REQUIRED_COLUMNS",
           "SuperScene", "LinkedSuperScene",
           "make_bounds", "bounds_vectors", "flux_bounds",
           "isophotal_radius", "kron_radius"]


REQUIRED_COLUMNS = ("ra", "dec", "rhalf",
                    "source_index", "is_active", "is_valid",
                    "n_iter", "n_patch")



class SuperScene:
    """An object that describes *all* sources in a scene.
    It contains methods for checking out regions, and checking
    them back in while updating their parameters and storing meta-information.

    It generates SuperScene coordinates for all sources (defined as arcseconds
    of latitude and longitude from a the median coordinates of the scene)
    stored in the `scene_coordinates` attribute and builds a KD-Tree based on
    these coordinates for fast lookup.

    A region can be checked out.  The default is to randomly choose a single
    valid source from the catalog and find all sources within some radius of
    that seed source.  The weighting logic for the seeds can be adjusted by
    over-writing the `seed_weight()` method.

    Sources in regions that are checked out have their sources unavailable for
    further checkouts, until they are checked back in, with new parameters
    """

    def __init__(self, sourcecat=None, bands=None,
                 statefile="superscene.fits",                       # disk locations
                 target_niter=200, maxactive_fraction=0.1,          # stopping criteria
                 maxactive_per_patch=20, nscale=3,                  # patch boundaries
                 boundary_radius=8., maxradius=6., minradius=1,     # patch boundaries
                 bounds_kwargs={}):

        self.statefilename = statefile
        self.bands = bands
        self.shape_cols = Galaxy.SHAPE_COLS
        if sourcecat is not None:
            self.bounds_kwargs = bounds_kwargs
            self.ingest(sourcecat, **bounds_kwargs)

        self.n_active = 0
        self.n_fixed = 0

        self.maxactive_fraction = maxactive_fraction
        self.target_niter = target_niter

        self.maxradius = maxradius
        self.minradius = minradius
        self.maxactive = maxactive_per_patch
        self.boundary_radius = boundary_radius
        self.nscale = 3

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.writeout()

    def writeout(self, filename=None):
        if filename is None:
            filename = self.statefilename
        fits.writeto(filename, self.sourcecat, overwrite=True)
        with open(filename.replace(".fits", "_log.json"), "w") as fobj:
            logs = dict(sourcelog=self.sourcelog, patchlog=self.patchlog)
            json.dump(logs, fobj)

    @property
    def sparse(self):
        frac = self.n_active * 1.0 / self.n_sources
        return frac < self.maxactive_fraction

    @property
    def undone(self):
        return np.any(np.abs(self.sourcecat["n_iter"]) < self.target_niter)

    @property
    def parameter_columns(self):
        return self.bands + self.shape_cols

    def ingest(self, sourcecat, **bounds_kwargs):
        """Set the catalog, make bounds array, and initialize covariance matricices
        """
        self.set_catalog(sourcecat)

        # --- build the KDTree ---
        self.kdt = cKDTree(self.scene_coordinates)

        # make the initial bounds catalog
        self.bounds_catalog = make_bounds(self.sourcecat, self.bands,
                                          shapenames=self.shape_cols,
                                          **bounds_kwargs)

        # make initial covariance matrices (identities)
        n_param = len(self.parameter_columns)
        self.covariance_matrices =  np.reshape(np.tile(np.eye(n_param).flatten(), self.n_sources),
                                                (self.n_sources, n_param, n_param))
        # Logs
        self.sourcelog = {}
        self.patchlog = []

    def set_catalog(self, sourcecat):
        """Set the sourcecat attribute to the given catalog, doing some checks
        and setting some useful values

        Parameters
        ----------
        sourcecat : structured ndarray of shape (n_sources,)
            A catalog of source parameters.  This is a structured array where
            the column names must include several specific column types.
        """
        for c in REQUIRED_COLUMNS:
            assert c in sourcecat.dtype.names, "required column {} is not present.".format(c)

        self.sourcecat = sourcecat
        self.n_sources = len(self.sourcecat)
        self.cat_dtype = self.sourcecat.dtype
        self.sourcecat["source_index"][:] = np.arange(self.n_sources)
        self.sourcecat["is_valid"][:] = True
        self.sourcecat["is_active"][:] = False

        # save the original catalog
        self.original = sourcecat.copy()

    def sky_to_scene(self, ra, dec):
        """Generate scene coordinates, which are anglular offsets (lat, lon)
        from the median ra, dec in units of arcsec.

        Parameters
        ----------
        ra : float or ndarray
            The right ascension (degrees)

        dec : float or ndarray
            The declination (degrees)

        Returns
        -------
        lat : float or ndarray
            The scene latitude coordinate of the input coordinates.  Arcsec.

        lon : float or ndarray
            The scene longitude coordinate of the input coordinates.  Arcsec.
        """
        c = SkyCoord(ra, dec, unit="deg")
        xy = c.transform_to(self.scene_frame)
        return xy.lon.arcsec, xy.lat.arcsec

    @property
    def scene_frame(self):
        """Generate and cache (or return cached version) of the scene frame,
        which is a lat-lon coordinate frame centered on the median RA and Dec
        of the sources.

        Returns
        -------
        frame : an astropy.corrdinates.Frame() instance
            The coordinate frame centered on the middle of the superscene.
        """
        try:
            return self._scene_frame
        except AttributeError:
            mra = np.median(self.sourcecat["ra"])
            mdec = np.median(self.sourcecat["dec"])
            center = SkyCoord(mra, mdec, unit="deg")
            self._scene_frame = center.skyoffset_frame()
            self._scene_center = (mra, mdec)
            return self._scene_frame

    @property
    def scene_coordinates(self):
        """Return cached scene coordinates for all sources, or, if not present,
        build the scene frame and generate and cache the scene coordinates
        before returning them.

        Returns
        -------
        scene_coordinates : ndarray of shape (n_source, 2)
            The scene coordinates.  These are given in arcseconds of latitude
            and longitude in a coordinate system centered on the median RA and
            Dec of the sources.
        """
        try:
            return self._scene_coordinates
        except(AttributeError):
            x, y = self.sky_to_scene(self.sourcecat["ra"],
                                     self.sourcecat["dec"])
            self.scene_x, self.scene_y = x, y
            self._scene_coordinates = np.array([self.scene_x, self.scene_y]).T
            return self._scene_coordinates

    def checkout_region(self, seed_index=-1):
        """Get a proposed region and the active and fixed sources that belong
        to that region.  Active sources are marked as such in the `sourcecat`
        and both active and fixed sources are marked as invalid for further
        patches.  The count of active and fixed sources is updated.

        Parameters
        ----------
        seed_index : int (optional)
            If >=0, use this (zero-indexed) source to seed the region.

        Returns
        -------
        region : A region.Region instance

        active : structured ndarray
            Copies of the rows of the `sourcecat` attribute corresponding to the
            active sources in the region

        fixed : structured ndarray
            Copies of the rows of the `sourcecat` attribute corresponding to the
            fixed sources in the region
        """
        # Draw a patch center, convert to scene coordinates
        # (arcsec from scene center), and get active and fixed sources
        cra, cdec = self.draw_center(seed_index=seed_index)
        center = self.sky_to_scene(cra, cdec)
        radius, active_inds, fixed_inds = self.get_circular_scene(center)
        # Deal with case that region is invalid
        if radius is None:
            return center, None, None
        region = CircularRegion(cra, cdec, radius / 3600.)
        self.sourcecat["is_active"][active_inds] = True
        self.sourcecat["is_valid"][active_inds] = False
        self.sourcecat["is_valid"][fixed_inds] = False
        self.n_active += len(active_inds)
        self.n_fixed += len(fixed_inds)

        return region, self.sourcecat[active_inds], self.sourcecat[fixed_inds]

    def checkin_region(self, active, fixed=None, niter=1, block_covs=None,
                       new_bounds=None, taskID=None, flush=False):
        """Check-in a set of active source parameters, and also fixed sources.
        The parameters of the active sources are updated in the master catalog,
        they are marked as inactive, and along with the provided fixed sources
        are marked as valid.  The counts of active and fixed sources are updated.
        The number of patches and iterations for each active source is updated.

        Parameters
        ----------
        active : structured array of shape (n_sources,)
            The final source parameters as a structured array.  Parameter names
            are in column names, one row per source.  Must include a
            "source_index" column.

        fixed : structured array of shape (n_sources,)
            The fixed sources for this patch.  Used to make them valid again

        niter : int
            The number of iterations that were run since last checkout.

        block_covs : ndarray of shape(n_sources,n_params, n_params)
            The covariance matrices for each source.

        taskID : int or string
            The ID of the task that did the sampling; used for logging purposes

        flush : bool
            If true, flush the new superscene catalog (the current parameter
            state) to disk, including patchlog
        """

        # Find where the sources are that are being checked in
        try:
            active_inds = active["source_index"].astype(int)
        except(KeyError):
            raise
        if fixed is not None:
            fixed_inds = fixed["source_index"].astype(int)
        else:
            fixed_inds = []

        # replace active source parameters with new parameters
        for f in self.parameter_columns:
            self.sourcecat[f][active_inds] = active[f]

        # update metadata
        self.sourcecat["n_iter"][active_inds] += niter
        self.sourcecat["is_active"][active_inds] = False
        self.sourcecat["n_patch"][active_inds] += int(niter > 0)
        self.sourcecat["is_valid"][active_inds] = True
        self.sourcecat["is_valid"][fixed_inds] = True

        self.n_active -= len(active_inds)
        self.n_fixed -= len(fixed_inds)

        # update bounds if they were changed during initialization
        if new_bounds is not None:
            self.bounds_catalog[active_inds] = new_bounds[:]

        # update mass matrix
        if block_covs is not None:
            # FIXME: this should account for missing bands in the mass matrices
            try:
                self.covariance_matrices[active_inds] = block_covs
            except(ValueError, AttributeError):
                print("could not update mass matrix")

        # log which patch and which child ran for each source?
        if taskID is not None:
            pid = str(taskID)  # JSON wants regular ints or str
            for k in active_inds:
                sid = int(k)
                if sid in self.sourcelog:
                    self.sourcelog[sid].append(pid)
                else:
                    self.sourcelog[sid] = [pid]
            self.patchlog.append(pid)

        if flush:
            self.writeout()

    def bounds_and_covs(self, sourceIDs):
        bounds = self.bounds_catalog[sourceIDs]

        if hasattr(self, "covariance_matrices"):
            covs = self.covariance_matrices[sourceIDs]
            cov = block_diag(*covs)
        else:
            cov = None

        return bounds, cov

    def reset(self):
        """Reset active, valid, and n_iter values.
        """
        self.sourcecat["is_valid"][:] = True
        self.sourcecat["is_active"][:] = False
        self.sourcecat["n_iter"][:] = 0
        self.sourcecat["n_patch"][:] = 0
        self.n_active = 0
        self.n_fixed = 0

    def get_circular_scene(self, center):
        """
        Parameters
        ----------
        center: 2-element array
            Central coordinates in scene units (i.e. arcsec from scene center)

        Returns
        -------
        radius: float
            The radius (in arcsec) from the center that encloses all active sources.

        active_inds: ndarray of ints
            The indices in the supercatalog of all the active sources

        fixed_inds: ndarray of ints
            The indices in the supercatalog of the fixed sources
            (i.e. sources that have some overlap with the radius but are not active)
        """
        # pull all sources within boundary radius
        # Note this uses original positions
        kinds = self.kdt.query_ball_point(center, self.boundary_radius)
        kinds = np.array(kinds)

        # check for active sources; if any exist, return None
        # really should do this check after computing a patch radius
        if np.any(self.sourcecat[kinds]["is_active"]):
            return None, None, None

        # sort sources by distance from center in scale-lengths
        # Note this uses original coordinates, but updated half-lengths
        rhalf = self.sourcecat[kinds]["rhalf"]
        d = self.scene_coordinates[kinds] - center
        distance = np.hypot(*d.T)
        # This defines a kind of "outer" distnce for each source
        # as the distance plus some number of half-light radii
        # TODO: should use scale radii? or isophotes?
        outer = distance + self.nscale * rhalf
        inner = distance - self.nscale * rhalf

        # Now we sort by outer distance.
        # TODO: *might* want to sort by just distance
        metric = outer
        order = np.argsort(metric)

        # How many sources have an outer distance within max patch size
        N_inside = (metric < self.maxradius).sum()
        # restrict to <= maxactive.
        N_active = min(self.maxactive, N_inside)

        # set up to maxsources active, add them to active scene
        #active = candidates[order][:N_active]
        active_inds = order[:N_active]
        finds = order[N_active:]
        # define a patch radius:
        # This is the max of active dist + Ns * rhalf, up to maxradius,
        # and at least 1 arcsec
        radius = outer[order][:N_active].max()
        radius = max(self.minradius, radius)

        # find fixed sources, add them to fixed scene:
        #   1) These are up to maxsources sources within Ns scale lengths of the
        #   patch radius defined by the outermost active source
        #   2) Alternatively, all sources within NS scale radii of an active source
        fixed_inds = finds[inner[finds] < radius][:min(self.maxactive, len(finds))]
        # FIXME: make sure at least one source is fixed?
        if len(fixed_inds) == 0:
            fixed_inds = finds[:1]
        return radius, kinds[active_inds], kinds[fixed_inds]

    def draw_center(self, seed_index=-1):
        """Randomly draw a center for the proposed patch.  Currently this
        works by drawing an object at random, with weights given by the
        `seed_weight` method.

        Parameters
        ----------
        seed_index : int or None (default, None)
             If non-zero int, override the random draw to pull a specific source.

        Returns
        -------
        ra : float
            RA of the center (decimal degrees)

        dec : float
            Declination of the center (decimal degrees)
        """
        if seed_index >= 0:
            k = seed_index
        else:
            k = np.random.choice(self.n_sources, p=self.seed_weight())
        seed = self.sourcecat[k]
        return seed["ra"], seed["dec"]

    def seed_weight(self):
        return self.exp_weight()

    def exp_weight(self):
        # just one for inactive, zero if active
        w = (~self.sourcecat["is_valid"]).astype(np.float)
        # multiply by exponential
        n = np.abs(self.sourcecat["n_iter"])
        mu = min(n.min(), self.target_niter)
        sigma = 20
        w *= np.exp((n.mean() - n) / sigma)
        return w / w.sum()

    def sigmoid_weight(self):
        # just one for inactive, zero if active
        w = (~self.sourcecat["is_active"]).astype(np.float)

        # multiply by something inversely correlated with niter
        # sigmoid ?  This is ~ 0.5 at niter ~ntarget
        # `a` controls how fast it goes to 0 after ntarget
        # `b` shifts the 0.5 weight left (negative) and right of ntarget
        a, b = 20., -1.0
        x = a * (1 - self.sourcecat["n_iter"] / self.target_niter) + b
        w *= 1 / (1 + np.exp(-x))

        return w / w.sum()


class LinkedSuperScene(SuperScene):
    """Similar to SuperScene but uses a friends-of-friends algorithm to group
    sources based on overlap of their radii of influence (ROI) when checking
    out regions.
    """

    def __init__(self, *args, buffer=0.3, roi=None, strict=False, **kwargs):
        """
        Extra Parameters
        ----------------
        buffer : float
            The extra padding, in arcsec, to add to the radius of the circle
            enclosing the active sources.

        roi : ndarray
            'Radius of influence' the radius (in arcsec) for each source to
            determine overlaps

        strict : bool
            If False, allow partial groups to fill remaining slots in the
            active scene.  Otherwise, leave empty slots if a group doesn't fit,
            unless it's the first group
        """
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.strict = strict
        if roi is None:
            try:
                self.roi = self.sourcecat["rhalf"]
            except(AttributeError):
                print("No ROI defined for sources")
                pass
        else:
            self.roi = roi

    def grow_source(self, index):
        """Find all sources linked to the original source. This uses a simple
        FoF algortihm with pair specific metric to identify friends

        Parameters
        ----------
        index : int
            Index of the source to use as the original source (i.e. the row in
            `sourcecat`)

        Returns
        -------
        members : list of ints
            A list of unique indices of all the friends of the original source,
            including the original source.
        """
        seeds = [index]
        members, leaves = [], []
        # find linked sources
        while True:
            members += seeds
            # find unique sub-sources that are not already members
            for s in seeds:
                #seed = self.original[s]
                #center = self.sky_to_scene(seed["ra"], seed["dec"])
                center = self.scene_coordinates[s]
                branches = self.find_overlaps(center, self.roi[s])
                buds = []
                [buds.append(b) for b in branches if b not in leaves]
                leaves += buds
            leaves = [l for l in leaves if (l not in members)]
            if len(leaves) == 0:
                break
            else:
                seeds = leaves
                leaves = []
        return members

    def grow_scene(self, seed_index, max_fixed=None):
        """Construct a scene while preserving associations.  This will add
        sources, including associated sources, to the scene in order of
        increasing distance from the seed source, until the maximum number of
        active sources is reached.

        Returns
        -------
        region : region.Region() instance or 2-tuple
            If a valid scene is identified, this is a CircularRegion (that need
            not be centered on the original source) encompassing all of these
            sources plus a buffer region.  Otherwise, a 2-tuple giving (ra, dec)
            of the propsoed seed source.

        active : structured ndarray
            The catalog of the active sources in the scene, including the seed
            source.

        fixed : structured ndarray
            The catalog of the fixed sources in the scene.
        """
        seed = self.original[seed_index]
        #center = self.sky_to_scene(seed["ra"], seed["dec"])
        center = self.scene_coordinates[seed_index]
        candidates = self.find_overlaps(center, self.boundary_radius, sort=True)
        # check for active sources; if any exist, return None
        if np.any(self.sourcecat[candidates]["is_active"]):
            return (seed["ra"], seed["dec"]), None, None

        # Get actives by adding sources and their associations
        active_inds = []
        for s in [seed_index] + candidates.tolist():
            if s in active_inds:
                continue
            n_avail = self.maxactive - len(active_inds)
            to_add = self.grow_source(s)
            if len(to_add) > n_avail:
                # Fill remaining slots if lax or no slots filled
                if (not self.strict) or (len(active_inds) == 0):
                    active_inds += to_add[:n_avail]
                break
            active_inds += to_add
        active = self.sourcecat[active_inds]

        # now redetermine the center and the radius
        cra, cdec = np.mean(active["ra"]), np.mean(active["dec"])
        center = self.sky_to_scene(cra, cdec)
        d = self.scene_coordinates[active_inds] - center
        radius = np.max(np.hypot(*d.T) + self.roi[active_inds] + self.buffer)
        radius = max(self.minradius, radius)
        region = CircularRegion(cra, cdec, radius / 3600.)

        # now look for fixed sources
        if max_fixed is None:
            max_fixed = self.maxactive
        inregion = self.find_overlaps(center, 1.2*radius, sort=True)
        fixed_inds = [i for i in inregion if i not in active_inds]
        end = min(len(fixed_inds), max_fixed)
        fixed_inds = fixed_inds[:end]

        return region, active_inds, fixed_inds

    def checkout_region(self, seed_index=-1, max_fixed=None):
        """Checkout a region based on a particular or randomly drawn scene;
        sources are added to the scene using a friends-of-friends algorithm.
        The final region is centered on the mean ra and dec of the active
        sources.

        Parameters
        ----------
        seed_index : int, optional (default: -1)
            If provided, grow the FoF scene using this source as the seed.
            Otherwise draw a source at random (weighted by `self.seed_weight`)

        max_fixed : int, optional
            If provided, only the closest `max_fixed` sources will be used in
            the fixed scene.  If not provided (default) then
            `maxactive_per_patch` will be used

        Returns
        -------
        region : a forcepho.regions.CircularRegion instance
            A region object describing the sky coordinates of the patch.

        active : structured ndarray
            The `sourcecat` entries for the active sources.  Will be `None` if
            the seed resulted in an invalid source.

        fixed : structured ndarray
            The sourcecat entries for the fixed sources in the scene.  Can be
            `None` if there were no fixed sources within the minimum radius.
        """
        if seed_index < 0:
            seed_index = np.random.choice(self.n_sources, p=self.seed_weight())

        region, active_inds, fixed_inds = self.grow_scene(seed_index, max_fixed)
        if active_inds is None:
            return (region, seed_index), None, None
        if len(fixed_inds) == 0:
            fixed = None
        else:
            fixed = self.sourcecat[fixed_inds]

        self.sourcecat["is_active"][active_inds] = True
        self.sourcecat["is_valid"][active_inds] = False
        self.sourcecat["is_valid"][fixed_inds] = False
        self.n_active += len(active_inds)
        self.n_fixed += len(fixed_inds)

        return region, self.sourcecat[active_inds], fixed

    def find_overlaps(self, center, radius, sort=False):
        """Find overlapping sources *not including at the provided center*
        Uses a metric based on |x_i - x_j|/|R_i + R_j|

        Parameters
        ----------
        center : ndarray of shape (2,)
            Floats representing the ra and dec of the center
            *in scene coordinates*

        radius : float
            Radius of influence of the `center`

        sort : bool, optional
            Whether to sort the output in increasing order of the metric

        Returns
        -------
        inds : ndarray of ints
            The indices of the sources that overlap, optionally sorted
            in order of distance from the center
        """
        kinds = self.kdt.query_ball_point(center, radius + self.boundary_radius)
        d = self.scene_coordinates[kinds] - center
        metric = np.hypot(*d.T) / (radius + self.roi[kinds])
        overlaps = (metric < 1) & (metric > 0)
        if sort:
            order = np.argsort(metric[overlaps])
        else:
            order = slice(None)
        return np.array(kinds)[overlaps][order]

    def overlap_circle(self, ra, dec, radius):
        """Get all sources with an roi that overlaps the given circle on the
        sky.

        Parameters
        ----------
        ra : float, degrees

        dec : degrees

        radius : circle radius in scene units, usually arcseconds (see
                 `sky_to_scene()` for details)
        """
        center = self.sky_to_scene(ra, dec)
        overlaps = self.find_overlaps(center, radius)
        return self.sourcecat[overlaps]

    def make_group_catalog(self):
        """Compute and assign each source to a group using the
        friends of friends algortihm.

        Returns
        -------
        groupID : ndarray of shape (n_sources,)
            The index of the FoF group to which the source belongs
            (starting at zero)
        """
        groupID = np.zeros(self.n_sources) - 1
        gind = 0
        for i in range(self.n_sources):
            if groupID[i] >= 0:
                continue
            assoc = self.grow_source(i)
            groupID[assoc] = gind
            gind += 1
        return groupID


def make_bounds(active, filternames, shapenames=Galaxy.SHAPE_COLS,
                n_sig_flux=5., dra=None, ddec=None, n_pix=2, pixscale=0.03,
                sqrtq_range=(0.4, 1.0), pa_range=(-0.6 * np.pi, 0.6 * np.pi),
                rhalf_range=(0.03, 0.3), sersic_range=(1., 5.)):
    """Make a catalog of upper and lower bounds for the parameters of each
    source. This catalog is a structured array with fields for each of the
    source parameters, each containing a 2-element array of the form (lower,
    upper).  Each row is a different source

    Parameters
    ----------
    active : structured ndarray of shape (n_source,)
        The source catalog, with appropriate column names

    filternames : list of strings
        The names of the columns corresponding to the flux parameters

    shapenames: list of strings (optional)
        The names of the columns corresponding to positional and shape parameters

    n_sig_flux : float (optional)
        The number of flux sigmas to set for the prior width

    dra : ndarray of shape (n_source,) or (1,) (optional)
        The delta in RA degrees to use for the prior width

    ddec : ndarray of shape (n_source,) or (1,) (optional)
        The delta in Dec degrees to use for the prior width

    n_pix : float (optional)
        The number of pixels to use for a prior width in RA and Dec

    pixscale : float, (optional)
        The size of each pixel, in arcsec
    """

    pm1 = np.array([-1., 1.])

    if dra is None:
        dra = n_pix * pixscale / 3600. / np.cos(np.deg2rad(active["ra"]))
    if ddec is None:
        ddec = np.array([n_pix * pixscale / 3600.])

    # Make empty bounds catalog
    colnames = filternames + shapenames
    cols = [("source_index", np.int32)] + [(c, np.float64, (2,))
                                           for c in colnames]
    dtype = np.dtype(cols)
    bcat = np.zeros(len(active), dtype=dtype)

    # Fill the easy ones
    bcat["q"] =  sqrtq_range
    bcat["pa"] = pa_range
    bcat["sersic"] = sersic_range
    bcat["rhalf"] = rhalf_range

    # Fill the ra, dec
    bcat["ra"] = active["ra"][:, None] + pm1[None, :] * dra[:, None]
    bcat["dec"] = active["dec"][:, None] + pm1[None, :] * ddec[:, None]

    # fill the fluxes
    for b in filternames:
        try:
            sigma_flux = active["{}_unc".format(b)]
            lo, hi = (pm1[None, :] * n_sig_flux * sigma_flux[:, None]).T
        except(ValueError, KeyError):
            #sigma_flux = np.sqrt(np.abs(active[b]))
            lo, hi = flux_bounds(active[b], n_sig_flux)
        bcat[b][:, 0] = lo
        bcat[b][:, 1] = hi
        #bcat[b] = active[b][:, None] + pm1[None, :] * n_sig_flux * sigma_flux[:, None]

    return bcat


def bounds_vectors(bounds_cat, filternames, shapes=Galaxy.SHAPE_COLS,
                   reference_coordinates=[0., 0.]):
    """Convert a structured array of bounds to simple 1d vectors of lower and
    upper bounds for all sources.
    """
    lower, upper = [], []
    bcat = bounds_cat.copy()
    bcat["ra"] -= reference_coordinates[0]
    bcat["dec"] -= reference_coordinates[1]
    for row in bcat:
        lower.extend([row[b][0] for b in filternames] + [row[c][0] for c in shapes])
        upper.extend([row[b][1] for b in filternames] + [row[c][1] for c in shapes])

    return np.array(lower), np.array(upper)


def flux_bounds(flux, unc1, snr_max=10):
    """Generate flux bounds based on initial guesses and a reference S/N.

    Parameters
    ----------
    flux : ndarray
        Array of object fluxes

    unc1 : float
        The half-width of the bound at flux=1  (i.e., Nsigma/SNR)

    snr_max : float
        Do not let bounds get narrower than 100/snr_max percent of the initial
        flux guess.
    """
    snr = np.sqrt(np.abs(flux)) / unc1
    #snr = np.minimum(snr, snr_max)

    fmin = (unc1 / 2.)**2
    noise = np.abs(flux) / snr
    lo = flux - noise
    hi = flux + noise

    lo[flux <= fmin] = np.minimum(lo[flux <= fmin], -fmin)
    #hi = np.hypot(hi, fmin)
    hi = np.maximum(hi, fmin)

    return lo, hi


def kron_radius(rhalf, sersic, rmax=None):
    k = gammaincinv(2 * sersic, 0.5)
    if rmax:
        x = k*(rmax / rhalf)**(1./sersic)
        c = gammainc(3 * sersic, x) / gammainc(2 * sersic, x)
    else:
        c = gamma(3 * sersic) / gamma(2 * sersic)
    r_kron = rhalf / k**sersic * c
    return r_kron


def I_eff(lum, rhalf, flux_radius=None, sersic=1):
    """gamma(2n, b_n) = 1/2; b_n = (re / r0)**(1/n)
    """
    two_n = 2 * sersic
    k = gammaincinv(two_n, 0.5)
    f =  gamma(two_n)
    if flux_radius is not None:
        x = k * (flux_radius / rhalf)**(1. / sersic)
        f *= gammainc(two_n, x)
    conv = k**(two_n) / (sersic * np.exp(k)) / f
    Ie = lum * conv / (2 * np.pi * rhalf**2)
    return Ie


def isophotal_radius(iso, flux, r_half, flux_radius=None, sersic=1):
    Ie = I_eff(flux, r_half, flux_radius=flux_radius, sersic=sersic)
    k = gammaincinv(2 * sersic, 0.5)
    r_iso = (1 - np.log(iso / Ie) / k)**(sersic)

    return r_iso * r_half


if __name__ == "__main__":

    catname = "../data/catalogs/hlf_xdf_v0.2.fits"
    raw = fits.getdata(catname)
    cat, bands, hdr = rectify_catalog(catname)

    # check you get unique sources
    sceneDB = LinkedSuperScene(cat, bands, roi=raw["rhalf"], buffer=0.5)
    k = np.argmax(sceneDB.roi)
    test = sceneDB.grow_source(k)
    assert len(test) == len(np.unique(test))

    sys.exit()
    # make a group catalog
    gid = sceneDB.make_group_catalog()