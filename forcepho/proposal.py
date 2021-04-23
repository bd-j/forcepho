# -*- coding: utf-8 -*-

"""proposal.py

This is the CPU-side interface to evaluate a likelihood on the GPU (i.e. make a
proposal).  So this is where we actually launch the CUDA kernels with PyCUDA.
The corresponding CUDA data model is in proposal.cu, and the CUDA kernels are
in compute_gaussians_kernel.cu.
"""

import sys
import os.path

import numpy as np

from . import source_dir
from .kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS

try:
    import pycuda
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except:
    pass


class Proposer:

    """
    This class invokes the PyCUDA kernel.
    It may also manage pinned memory in a future version.
    """

    def __init__(self, patch, thread_block=1024, shared_size=48000,
                 max_registers=64, show_ptxas=False, debug=False,
                 kernel_name='EvaluateProposal', chi_dtype=np.float64,
                 kernel_fn='compute_gaussians_kernel.cu'):

        self.grid = (patch.n_bands, 1)
        self.block = (thread_block, 1, 1)
        self.shared_size = shared_size
        self.chi_dtype = chi_dtype
        self.patch = patch

        thisdir = os.path.abspath(os.path.dirname(__file__))

        with open(os.path.join(source_dir, kernel_fn), 'r') as fp:
            kernel_source = fp.read()

        options = ['-std=c++11']

        #if chi_dtype is np.float64:
        #    options += ["-arch sm_70"]

        if show_ptxas:
            options += ['--ptxas-options=-v']

        if debug:
            options += ['-O0', '--ptxas-options=-O0', '-lineinfo']

        if max_registers:
            options += ['-maxrregcount={}'.format(max_registers)]

        mod = SourceModule(kernel_source, cache_dir=False,
                           include_dirs=[source_dir, thisdir],
                           options=options, no_extern_c=True)
        self.evaluate_proposal_kernel = mod.get_function(kernel_name)

    def evaluate_proposal(self, proposal, verbose=False, unpack=True):
        """Call the GPU kernel to evaluate the likelihood of a parameter proposal.

        Parameters
        ----------
        proposal: ndarray of dtype `source_struct_dtype`
            An array of source parameters, packed into a Numpy array
            (and thus ready to send to the GPU).

        Returns
        -------
        chi2: float
            The chi^2 for this proposal

        chi2_derivs: ndarray of dtype `source_float_dt`
            The derivatives of chi^2 with respect to proposal parameters.
            This is an array with shape (nband, nactive_sources, 7)

        residuals: list of ndarray of shape of original exposures
            The residual image (data - model) for each exposure.  No padding.
            Only returned if patch.return_residual.
        """

        chi_out = np.empty(self.patch.n_bands, dtype=self.chi_dtype)
        chi_derivs_out = np.empty(self.patch.n_bands,
                                  dtype=response_struct_dtype)

        if verbose:
            msg = "Launching with grid {}, block {}, shared {}"
            msg.format(self.grid, self.block, self.shared_size)
            print(msg, file=sys.stderr, flush=True)
        # is this synchronous?
        # do we need to "prepare" the call?
        self.evaluate_proposal_kernel(self.patch.gpu_patch, cuda.In(proposal),     # inputs
                                      cuda.Out(chi_out), cuda.Out(chi_derivs_out), # outputs
                                      grid=self.grid, block=self.block,            # launch config
                                      shared=self.shared_size)

        # Reshape the output
        vshape = self.patch.n_bands, MAXSOURCES, NPARAMS
        chi_derivs_out = chi_derivs_out.view(np.float32).reshape(vshape)
        chi_derivs_out = chi_derivs_out[:, :self.patch.n_sources]

        # Unpack residuals
        if self.patch.return_residual:
            residuals = self.retrieve_array("residual")
            if unpack:
                residuals = self.unpack_residuals(residuals)

        if self.patch.return_residual:
            return chi_out, chi_derivs_out, residuals
        else:
            return chi_out, chi_derivs_out

    def retrieve_array(self, arrname):
        flatdata = cuda.from_device(self.patch.cuda_ptrs[arrname],
                                    shape=self.patch.xpix.shape,
                                    dtype=self.patch.pix_dtype)
        return flatdata

    def unpack_residuals(self, residuals_flat, reshape=False):
        """Unpack flat, padded residuals into original images
        """
        residuals = np.split(residuals_flat, np.cumsum(self.patch.exposure_N)[:-1])

        # This tries to reshape the residuals into square stamps after removing
        # padding, if that's how the data was originally packed.  Otherwise, one
        # would want have the xpix and ypix arrays along with the residuals to
        # be able to reconstruct an image
        if reshape:
            for e, residual in enumerate(residuals):
                residual = residual[:self.patch.original_sizes[e]]
                residuals[e] = residual.reshape(self.patch.original_shapes[e])

        return residuals


source_float_dt = np.float32
source_struct_dtype = np.dtype([('ra', source_float_dt),
                                ('dec', source_float_dt),
                                ('q', source_float_dt),
                                ('pa', source_float_dt),
                                ('nsersic', source_float_dt),
                                ('rh', source_float_dt),
                                ('fluxes', source_float_dt, MAXBANDS),
                                ('mixture_amplitudes', source_float_dt, MAXRADII),
                                ('damplitude_drh', source_float_dt, MAXRADII),
                                ('damplitude_dnsersic', source_float_dt, MAXRADII)],
                               align=True)

response_float_dt = source_float_dt
response_struct_dtype = np.dtype([('dchi2_dparam', response_float_dt,
                                   NPARAMS * MAXSOURCES)], align=True)
