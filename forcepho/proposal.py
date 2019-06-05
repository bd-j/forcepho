'''
proposal.py

This is the CPU-side interface to evaluate a likelihood on the GPU
(i.e. make a proposal).  So this is where we actually launch the CUDA
kernels with PyCUDA.  The corresponding CUDA data model is
in proposal.cu, and the CUDA kernels are in compute_gaussians_kernel.cu.
'''

import numpy as np

from .kernel_limits import MAXBANDS, MAXRADII

import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class Proposer:
    '''
    This class invokes the PyCUDA kernel.
    It may also manage pinned memory in a future version.
    '''

    def __init__(self, patch, thread_block=1024, kernel_name='EvaluateProposal', kernel_fn='compute_gaussians_kernel.cu'):

        self.grid = (patch.n_bands,1)
        self.block = (thread_block,1,1)

        mod = SourceModule(kernel_fn)
        self.kernel = mod.get_function(kernel_name)


    def evaluate_proposal(self, proposal):
        # TODO: use pinned memory here?
        proposal_allocation = cuda.to_device(proposal)
        proposal_cuda_ptr = np.uintp(proposal_allocation)

        # alloc responses

        # is this synchronous?
        # do we need to "prepare" the call?
        kernel(patch, proposal_cuda_ptr, grid=self.grid, block=self.block)


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
