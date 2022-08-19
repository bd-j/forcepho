# -*- coding: utf-8 -*-

"""io.py - utilities for writing patch & chain data to disk
"""

import numpy as np
import json
import h5py

__all__ = ["write_to_disk", "write_residuals"]


def write_to_disk(out, outroot, model, config, residual=None):
    """Write chain and optionally the residuals for a patch to disk.

    Parameters
    ----------
    out : dictionary
        The filled output object from one of the fitting methods

    outroot : string
        The base path and filename for the output.  Chain and residuals will be
        written to `<outroot>_samples.h5` and `<outroot>_residuals.h5`

    model : an instance of forcepho.model.Posterior()
        Model object used for constructing residuals if not already supplied

    config : Namespace
        Configuration parameters used for the run.  The value of
        `config.write_residuals` will be used to decide whether to write a
        residual object as well.

    residual : optional, list of ndarray
        The residual pixel values in each band.  If not given, the residuals
        will be computed from the last step in the chain.
    """
    # --- write the chain and meta-data for this task ---
    outfile = f"{outroot}_samples.h5"
    try:
        out.config = json.dumps(vars(config))
    except(TypeError):
        pass
    out.dump_to_h5(outfile)

    # --- Write image data and residuals if requested ---
    if getattr(config, "write_residuals", False):
        outfile = f"{outroot}_residuals.h5"
        if residual is None:
            q = out.chain[-1, :]  # last position in chain
            residual = model.residuals(q)
        write_residuals(model.patch, outfile, residuals=residual)


def write_residuals(patcher, filename, residuals=None):
    # TODO: Should this be a method on Patch or Posterior ?
    pixattrs = ["data", "xpix", "ypix", "ierr"]
    metas = ["D", "CW", "crpix", "crval"]
    epaths = patcher.epaths

    with h5py.File(filename, "w") as out:
        out.create_dataset("epaths", data=np.array(epaths, dtype="S"))
        out.create_dataset("ebands", data=np.array(patcher.bands, dtype="S"))
        #out.create_dataset("exposure_start", data=patcher.exposure_start)
        out.create_dataset("reference_coordinates", data=patcher.patch_reference_coordinates)
        #out.create_dataset("active", data=)
        #out.create_dataset("fixed", data=fixed)

        for band in patcher.bandlist:
            g = out.create_group(band)

        #residual = np.split(model._residuals, np.cumsum(patcher.exposure_N)[:-1])
        if residuals is not None:
            make_imset(out, epaths, "residual", residuals)

        for a in pixattrs:
            arr = patcher.split_pix(a)
            make_imset(out, epaths, a, arr)

        for a in metas:
            arr = getattr(patcher, a)
            make_imset(out, epaths, a, arr)


def make_imset(out, paths, name, arrs):
    for i, epath in enumerate(paths):
        try:
            g = out[epath]
        except(KeyError):
            g = out.create_group(epath)

        try:
            g.create_dataset(name, data=np.array(arrs[i]))
        except:
            print("Could not make {}/{} dataset from {}".format(epath, name, arrs[i]))
