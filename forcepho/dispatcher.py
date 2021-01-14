#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""dispatcher.py

Parent side classes for MPI queues and handling a master source catalog,
where sources/patches are checked out and checked back in
"""

import os
import json
import logging
import argparse
import time
from argparse import Namespace
import socket

import numpy as np

from scipy.spatial import cKDTree
from scipy.linalg import block_diag
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from .region import CircularRegion
from .sources import Galaxy
from .fitting import run_lmc
from .model import GPUPosterior, BoundedTransform
from .utils import rectify_catalog, read_config


REQUIRED_COLUMNS = ("ra", "dec", "rhalf",
                    "source_index", "is_active", "is_valid",
                    "n_iter", "n_patch")

# TODO: does setting up logging here conflict with other modules' use of logger?
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dispatcher')


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

    Sources in regions that are checked out have their sources unavailable for further
    checkouts, until they are checked back in, with new parameters
    """

    def __init__(self, statefile="superscene.fits",                 # disk locations
                 target_niter=200, maxactive_fraction=0.1,          # stopping criteria
                 maxactive_per_patch=20, nscale=3,                  # patch boundaries
                 boundary_radius=8., maxradius=5., minradius=1,     # patch boundaries
                 sourcecat=None, bands=None, bounds_kwargs={}):

        self.statefilename = statefile
        self.bands = bands
        self.shape_cols = Galaxy.SHAPE_COLS
        if sourcecat is not None:
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

        # --- build the KDTree ---
        self.kdt = cKDTree(self.scene_coordinates)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.writeout()

    def writeout(self):
        fits.writeto(self.statefilename, self.sourcecat, overwrite=True)
        with open(self.statefilename.replace(".fits", "_log.json"), "w") as fobj:
            logs = dict(sourcelog=self.sourcelog, patchlog=self.patchlog)
            json.dump(logs, fobj)

    @property
    def sparse(self):
        frac = self.n_active * 1.0 / self.n_sources
        return frac < self.maxactive_fraction

    @property
    def undone(self):
        return np.any(self.sourcecat["n_iter"] < self.target_niter)

    @property
    def parameter_columns(self):
        return self.bands + self.shape_cols

    def ingest(self, sourcecat, **bounds_kwargs):
        """Set the catalog, make bounds array, and initialize covariance matricices
        """
        self.set_catalog(sourcecat)
        self.bounds_catalog = make_bounds(self.sourcecat, self.bands,
                                          shapenames=self.shape_cols,
                                          **bounds_kwargs)
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

    def checkin_region(self, active, fixed, niter, block_covs=None,
                       taskID=None, flush=False):
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
            fixed_inds = fixed["source_index"].astype(int)
        except(KeyError):
            raise

        # replace active source parameters with new parameters
        for f in self.parameter_columns:
            self.sourcecat[f][active_inds] = active[f]

        # update metadata
        self.sourcecat["n_iter"][active_inds] += niter
        self.sourcecat["is_active"][active_inds] = False
        self.sourcecat["n_patch"][active_inds] += 1
        self.sourcecat["is_valid"][active_inds] = True
        self.sourcecat["is_valid"][fixed_inds] = True

        self.n_active -= len(active_inds)
        self.n_fixed -= len(fixed_inds)

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
        w = (~self.sourcecat["is_active"]).astype(np.float)
        n = self.sourcecat["n_iter"]
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

    def get_grown_scene(self):
        # option 1
        # grow the tree.
        # stopping criteria:
        #    you hit an active source;
        #    you hit maxactive;
        #    no new sources within tolerance

        # add fixed boundary objects; these will be all objects
        # within some tolerance (d / size) of every active source
        raise NotImplementedError


def make_bounds(active, filternames, shapenames=Galaxy.SHAPE_COLS,
                n_sig_flux=5., dra=None, ddec=None, n_pix=2, pixscale=0.03,
                sqrtq_range=(0.3, 1.0), pa_range=(-0.6 * np.pi, 0.6 * np.pi),
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

    # Fill the ra
    bcat["ra"] = active["ra"][:, None] + pm1[None, :] * dra[:, None]
    bcat["dec"] = active["dec"][:, None] + pm1[None, :] * ddec[:, None]

    # fill the fluxes
    for b in filternames:
        try:
            sigma_flux = active["{}_unc".format(b)]
        except(ValueError):
            sigma_flux = np.sqrt(np.abs(active[b]))
        bcat[b] = active[b][:, None] + pm1[None, :] * n_sig_flux * sigma_flux[:, None]

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


class MPIQueue:
    """Simple implementation of an MPI queue.  Work can be submitted to the queue
    as long as there is at least one idle child.  Supplies an interface to collect
    all returned work units.

    Implementation note: if we encapsulate any references to MPI in the function
    bodies, we can avoid a top-level MPI import.

    Parameters
    ----------
    comm : An MPI communicator

    n_children : int
        The number of child processes

    Attributes
    ----------
    busy : A list of 2-tuples
        Each tuple gives the child id and the request object for the submitted
        task

    idle : list of ints
        The idle children. This is initialized to be all children.
    """

    def __init__(self, comm, n_children):

        self.comm = comm
        self.irecv_handles = []
        self.busy = []
        self.idle = list(range(1,n_children + 1))
        self.n_children = n_children
        self.parent = 0

        # TODO is this the right place for this?
        self.ntry = 1000
        self.irecv_bufsize = int(10e6)

    def collect(self, blocking=False):
        """Collect all results that children have finished.

        Parameters
        ----------
        blocking: bool or str, optional
            Whether to wait for at least one result.
            May be 'all' to force waiting for all results.
            Default: False

        Returns
        -------
        result : object
            The result generated and passed by the child.
        """
        if blocking == 'all' and not self.irecv_handles:
            return []

        if blocking and not self.irecv_handles:
            raise RuntimeError('blocking collect() requested but there are no outstanding tasks')

        statuses = [MPI.Status() for _ in range(len(self.irecv_handles))]
        if blocking == 'all':
            results = MPI.Request.waitall(self.irecv_handles, statuses)
            indices = range(len(self.irecv_handles))
        elif blocking:
            # lowercase waitsome/testsome require bleeding-edge mpi4py
            indices, results = MPI.Request.waitsome(self.irecv_handles, statuses)
        else:
            indices, results = MPI.Request.testsome(self.irecv_handles, statuses)

        for i in sorted(indices, reverse=True):
            # TODO: this happens at Flatiron sometimes. Bad MPI? Bad mpi4py? mpi4py bug?
            # Problem with irecv allocation? Incomplete communication?
            # The results look sane though...
            if statuses[i].source == -1:
                logger.warning(f'Got source rank -1 on irecv from rank {self.busy[i]}!  Why??')
            self.idle += [self.busy.pop(i)]
            del self.irecv_handles[i]

        logger.info(f'Collected {len(results)} result(s) from child(ren) {self.idle[len(self.idle)-len(results):]}')

        assert len(self.idle) == len(set(self.idle))  # check uniqueness

        return results

    def submit(self, task, tag='any'):
        """Submit a single task to the queue.  This will be assigned to the
        child process at the top of the (idle) queue. If no children are idle,
        an error is raised.
        """
        if tag == 'any':
            tag = MPI.ANY_TAG

        # Why make this an error rather than queue it up for later?
        # Because we want to generate tasks with the freshest SuperScene info
        try:
            child = self.idle.pop(0)
        except IndexError as e:
            raise RuntimeError('No idle child to submit task to!') from e

        # We'll use a blocking send for this, since the child ought to be listening,
        # as good children do.
        # And then we don't have to worry about the lifetime of the task or isend handle.
        logger.info(f'Sending task {tag} to child {child}')
        if child in self.busy:
            raise ValueError(f'About to send task to child {child} that is already busy!')
        self.comm.send(task, dest=child, tag=tag)
        # Queue up the eventual receive of the result
        irecv_bufsize = int(10e6)  # TODO: have to specify a max recv size. Better way to do this? The Jades demo payload is 50 KB.
        rreq = self.comm.irecv(irecv_bufsize, source=child, tag=tag)
        self.irecv_handles += [rreq]
        self.busy += [child]
        return child

    def closeout(self):
        """Send kill messages (`None`) to all the children.
        """
        if len(self.idle) != self.n_children:
            raise RuntimeError('Trying to close MPIQueue, but some children are busy!')
        if len(self.irecv_handles) != 0:
            raise RuntimeError(f'Trying to close MPIQueue, but have {len(self.irecv_handles)} outstanding task(s)!')

        logger.info(f'Poisoning children')

        for child in range(1,self.n_children + 1):
            self.comm.send(None, dest=child, tag=0)


def do_parent(comm, config=None):
    timer = time.perf_counter
    tstart = timer()

    # TODO: not really sure if this is the "right" way to use logging
    # but isn't the point that we don't have to pass around a logger object?
    global logger
    logger = logging.getLogger('dispatcher-parent')
    logger.info(f'Starting parent on {socket.gethostname()}')

    rank = comm.Get_rank()
    n_child = comm.Get_size() - 1
    patchcat = {}

    # Make Queue
    queue = MPIQueue(comm, n_child)

    # --- Get the patch dispatcher ---  (parent)
    cat, bands, chdr = rectify_catalog(config.raw_catalog)
    sceneDB = SuperScene(sourcecat=cat, bands=bands,
                         maxactive_per_patch=config.maxactive_per_patch,
                         maxradius=config.patch_maxradius,
                         target_niter=config.target_draws,
                         statefile=config.scene_catalog,
                         bounds_kwargs={},
                         maxactive_fraction=0.5)
    checkout_time = 0

    # Do it in context so failure still writes current superscene.
    with sceneDB:
        logger.debug(f'SuperScene has {sceneDB.n_sources} sources')

        taskid = 0
        # Event Loop
        while sceneDB.undone:
            # Generate one patch proposal.
            _tstart = timer()
            ntry = getattr(config, "ntry_checkout", 1000)  # how many tries before we decide there are no regions to be checked out?
            for _ in range(ntry):
                region, active, fixed = sceneDB.checkout_region()
                if active is not None:
                    break
            else:
                logger.debug(f'Failed to checkout region')
            checkout_time += timer() - _tstart

            # Construct the task.
            if active is not None:
                if not sceneDB.sparse:
                    logger.debug(f'Scene no longer sparse with {queue.n_children - len(queue.idle) + 1} checkout(s)')
                logger.info(f'Checked out region with {len(active)} actives')
                taskid += 1
                bounds, cov = sceneDB.bounds_and_covs(active["source_index"])

                chore = {'region': region, 'active': active, 'fixed': fixed,
                         'bounds': bounds, 'cov': cov,
                         'bands': bands, 'shape_cols': sceneDB.shape_cols}
                # log some info for this patch/task
                patchcat[taskid] = {"ra": region.ra, "dec": region.dec, "radius": region.radius,
                                    "sources": active["source_index"].tolist()}
                # submit the task
                assigned_to = queue.submit(chore, tag=taskid)

            # Collect all results that have been returned.
            # If all workers are busy, or the scene is no longer sparse, or no regions
            # are available for checkout, wait for a result to come back
            blocking = not queue.idle or not sceneDB.sparse or active is None
            results = queue.collect(blocking=blocking)

            # Check results back in
            for result in results:
                sceneDB.checkin_region(result['final'], result['out'].fixed,
                                       len(result['out'].chain),
                                       block_covs=result['covs'],
                                       taskID=taskid)

        # Receive any stragglers
        results = queue.collect(blocking='all')
        logger.info(f'Collected {len(results)} straggler result(s)')
        # TODO: finalize the checkin args, then update this
        #for result in results:
        #    sceneDB.checkin_region(result.active, result.fixed,
        #                           result.niter, mass_matrix=None)

        ttotal = timer() - tstart
        logger.info(f"Finished in {ttotal:.1f}s")
        logger.info(f"Spent {checkout_time:.1f}s checking out sources")

    with open("patchlog.dat", "w") as f:
        json.dump(patchcat, f)
    queue.closeout()


def do_child(comm, config=None):
    rank = comm.Get_rank()
    global logger
    logger = logging.getLogger(f'dispatcher-child-{rank}')
    parent = 0

    # --- Patch Maker (gets reused between patches) ---
    from .patches import JadesPatch
    patcher = JadesPatch(metastore=config.metastorefile,
                            psfstore=config.psfstorefile,
                            pixelstore=config.pixelstorefile,
                            splinedata=config.splinedatafile,
                            return_residual=True)

    # --- Event Loop ---
    while True:
        status = MPI.Status()
        # do a blocking receive
        task = comm.recv(source=parent, tag=MPI.ANY_TAG,
                         status=status)
        logger.info(f'Received task id {status.tag}')
        # if shutdown break and quit
        if task is None:
            logger.info(f"Child {rank} shutting down.")
            break

        # To be explicit, let's unpack all the task variables here
        region, active, fixed = task['region'], task['active'], task['fixed']
        bounds, cov = task['bounds'], task['cov']
        bands, shape_cols = task['bands'], task['shape_cols']
        del task

        taskid = status.tag
        logger.info(f"Child {rank} received RA {region.ra}, DEC {region.dec} with tag {taskid}")

        # --- get pixel data and metadata, subtract fixed sources, build model ---
        patcher.build_patch(region, None, allbands=bands)
        model, q = patcher.prepare_model(active=active, fixed=fixed, bounds=bounds,
                                         shapes=shape_cols)
        logger.info("Prepared patch and model")

        # --- Sample using covariances--- (child)
        weight = max(10, active["n_iter"].min())
        logger.info(f"sampling with covariance weight={weight}")
        out, step, stats = run_lmc(model, q.copy(), n_draws=config.sampling_draws, warmup=config.warmup,
                                   z_cov=cov, full=config.full_cov, adapt=True,
                                   weight=weight, progressbar=getattr(config, "progressbar", False))

        # --- develop the payload ---
        logger.info("Sampling complete, preparing output.")
        final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                               step=step, stats=stats, patchID=taskid)
        payload = dict(out=out, final=final, covs=covs)

        # --- write the output for this task ---
        if config.patch_dir:
            outfile = os.path.join(config.patch_dir, "task{}_results.h5".format(taskid))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            logger.info("Writing to {}".format(outfile))
            out.dump_to_h5(outfile)


        # --- blocking send to parent, free GPU memory ---
        comm.send(payload, parent, status.tag)
        logger.info(f"Child {rank} sent {region.ra} for patch {taskid}")

        patcher.free()

    del patcher


def dummy_work(region, active, fixed):
    """Pretend to do work, but sleep for 1 second

    Returns
    -------
    result : Namespace() instance
        A simple namespace object with niter=75
    """
    time.sleep(1)
    result = Namespace()
    result.niter = 75
    result.active = active
    result.fixed = fixed
    return result


def main(config=None):
    # Demo usage of the queue

    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n_child = comm.Get_size() - 1

    if n_child < 1:
        logger.warning('Need at least one child rank!')

    if rank == 0:
        # Do parental things
        do_parent(comm, config=config)
    else:
        # Do childish things
        do_child(comm, config=config)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    # Hide this MPI import in __main__ so one can import this module without MPI
    from mpi4py import MPI

    parser = argparse.ArgumentParser(description='MPI dispatcher interface', formatter_class=ArgParseFormatter)
    # Any command line arguments can be added here
    parser.add_argument("--config_file", type=str, default="galsim.yml")

    args = parser.parse_args()
    config = read_config(args.config_file, args)

    main(config)
