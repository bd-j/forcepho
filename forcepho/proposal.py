'''
proposal.py

This is the CPU-side interface to evaluate a likelihood on the GPU
(i.e. make a proposal).  So this is where we actually launch the CUDA
kernels with PyCUDA.  The corresponding CUDA data model is
in proposal.cu, and the CUDA kernels are in compute_gaussians_kernel.cu.
'''

import sys
import os.path

import numpy as np

from .kernel_limits import MAXBANDS, MAXRADII, MAXSOURCES, NPARAMS

import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class Proposer:
    '''
    This class invokes the PyCUDA kernel.
    It may also manage pinned memory in a future version.
    '''

    def __init__(self, patch, thread_block=1024, shared_size=48000, max_registers=64, show_ptxas=False,
                    kernel_name='EvaluateProposal', kernel_fn='compute_gaussians_kernel.cu'):

        self.grid = (patch.n_bands,1)
        self.block = (thread_block,1,1)
        self.shared_size = shared_size
        self.patch = patch

        thisdir = os.path.abspath(os.path.dirname(__file__))

        with open(os.path.join(thisdir, kernel_fn), 'r') as fp:
            kernel_source = fp.read()

        options = ['-std=c++11', '-lineinfo']

        if show_ptxas:
            options += ['--ptxas-options=-v']

        if max_registers:
            options += [f'-maxrregcount={max_registers}']

        mod = SourceModule(kernel_source, cache_dir=False, include_dirs=[thisdir],
            options=options, no_extern_c=True)
        self.evaluate_proposal_kernel = mod.get_function(kernel_name)

    def evaluate_proposal(self, proposal):
        '''
        Call the GPU kernel to evaluate the likelihood of a parameter proposal.

        Parameters
        ----------
        proposal: ndarray of dtype `source_struct_dtype`
            An array of source parameters, packed into a Numpy array
            (and thus ready to send to the GPU).

        Returns
        -------
        chi2: float
            The chi^2 for this proposal

        chi2_derivs: ndarray of dtype `response_struct_dtype`
            The derivatives of chi^2 with respect to proposal parameters.
            Length of the array is nbands; each element is 7*MAXSOURCES floats.

        residuals: list of ndarray of shape of original exposures
            The residual image (data - model) for each exposure.  No padding.
            Only returned if patch.return_residual.

        '''

        chi_out = np.empty(1, dtype=np.float32)
        chi_derivs_out = np.empty(self.patch.n_bands, dtype=response_struct_dtype)

        print(f'Launching with grid {self.grid}, block {self.block}, shared {self.shared_size}',
            file=sys.stderr, flush=True)
        # is this synchronous?
        # do we need to "prepare" the call?
        self.evaluate_proposal_kernel(self.patch.gpu_patch, cuda.In(proposal),              # inputs
                                      cuda.Out(chi_out), cuda.Out(chi_derivs_out),  # outputs
                                      grid=self.grid, block=self.block, shared=self.shared_size,           # launch config
                                      )

        # Is this the right shape?
        chi_derivs_out = chi_derivs_out.view(np.float32).reshape(self.patch.n_bands, MAXSOURCES, 7)
        chi_derivs_out = chi_derivs_out[:,:self.patch.n_sources]

        # Unpack return values
        if self.patch.return_residual:
            residuals_flat = cuda.from_device(self.patch.cuda_ptrs['residual'], shape=self.patch.xpix.shape, dtype=self.patch.pix_dtype)
            residuals = self.unpack_residuals(residuals_flat)

        print(chi_out, chi_derivs_out)

        if self.patch.return_residual:
            return chi_out, chi_derivs_out, residuals
        else:
            return chi_out, chi_derivs_out


    def unpack_residuals(self, residuals_flat):
        # Unpack flat, padded residuals into original images
        residuals = np.split(residuals_flat, np.cumsum(self.patch.exposure_N)[:-1])
        
        for e,residual in enumerate(residuals):
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
                                ('damplitude_dnsersic', source_float_dt, MAXRADII),],
                                align=True)

respose_float_dt = source_float_dt
response_struct_dtype = np.dtype([('dchi2_dparam', respose_float_dt, NPARAMS*MAXSOURCES)], align=True)
