#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""dispatcher.py

Parent side classes for MPI queues.
"""

import os
import json
import logging
import argparse
import time
from argparse import Namespace
import socket

import numpy as np
from astropy.io import fits

from .superscene import SuperScene, LinkedSuperScene, rectify_catalog
from .utils import read_config
#from .child import sampling_task, optimization_task


__all__ = ["MPIQueue",
           "do_child", "do_parent"]

# TODO: does setting up logging here conflict with other modules' use of logger?
#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger('dispatcher')


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
                                       config.sampling_draws,
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
    logger = logging.getLogger(f'child-{rank}')
    parent = 0

    # --- Patch Maker (gets reused between patches) ---
    from .patches import JadesPatch
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)

    # --- What are we doing? ---
    #if config.mode == "optimization":
    #    work = optimization_task
    #elif config.mode == "sampling":
    #    work = sampling_task
    #else:
    work = dummy_work

    # --- Event Loop ---
    while True:
        status = MPI.Status()
        # do a blocking receive
        task = comm.recv(source=parent, tag=MPI.ANY_TAG,
                         status=status)
        logger.info(f'Child {rank} received task id {status.tag}')
        # if shutdown break and quit
        if task is None:
            logger.info(f"Child {rank} shutting down.")
            break

        answer = work(patcher, task, config, logger)

        # --- blocking send to parent, free GPU memory ---
        comm.send(answer, parent, status.tag)
        logger.info(f"Child {rank} sent answer for task id {status.tag}")

        patcher.free()

    del patcher


def dummy_work(patcher, task, config, logger):
    """Pretend to do work, but sleep for 1 second

    Returns
    -------
    result : Namespace() instance
        A simple namespace object with niter=75
    """
    logger.info("Sleeping for a second")
    time.sleep(1)
    result = Namespace()
    result.niter = 75
    result.active = task["active"]
    result.fixed = task["fixed"]
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
