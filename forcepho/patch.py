#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
patch.py -- data model for a patch on the sky

The corresponding GPU-side CUDA struct is in patch.cu.
"""

import numpy as np
import h5py

import pycuda
import pycuda.autoinit  # safe?
import pycuda.driver as cuda

class Patch:

    def __init__(self,
        stamps,            # A list of PostageStamp objects (exposure data) from FITS files
        miniscene,         # All peaks identified in this patch region
        mask=None,              # The mask that defines the nominal patch
        super_pixel_size=1,  # Number of pixels in each superpixel
        pix_dtype=np.float32,  # data type for pixel and flux data
        meta_dtype=np.float32   # data type for non-pixel data
        ):
        '''
        Constructs a Patch from PostageStamps (exposures) and a MiniScene
        (a list of pre-identified peaks/sources).  The Patch packs the
        exposures and sources in a manner suitable for sending to the GPU.
        This includes rearranging the data into (padded) super-pixel order.

        The Patch object contains methods for sending the patch data to
        the GPU with PyCUDA.

        Parameters
        ----------
        pix_dtype: np.dtype, optional
            The Numpy datatype of the pixel data, like fluxes and coordinates.
            Default: np.float32

        meta_dtype: np.dtype, optional
            The Numpy datatype of the non-pixel data, like transformation matrices.
            Default: np.float32
        '''

        self.pix_dtype = pix_dtype
        self.meta_dtype = meta_dtype

        # Group stamps by band
        # stamp.band must be an int identifier (does not need to be contiguous)
        band_ids = [st.band for st in stamps]
        assert (np.diff(band_ids) >= 0).all(), 'Stamps must be sorted by band'

        bands = [stamp.band for stamp in stamps]
        uniq_bands, n_exp_per_band = np.unique(bands, return_counts=True)

        # Use the stamps and miniscene to populate these
        self.n_bands = len(uniq_bands)          # Number of bands/filters
        self.n_exp = len(stamps)           # Number of exposures covering the patch (all bands)

        # Pixel Data
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        
        # Pack the 2D stamps into 1D arrays
        #self.pack_pix(stamps, mask, super_pixel_size)

        self.pack_source_metadata(miniscene)

        #self.pack_astrometry(miniscene.sources)

        #self.pack_psf(miniscene)


    def pack_psf(self, miniscene, dtype=None):
        '''
        Each Sersic radius bin has a number of Gaussians associated with it from the PSF.
        The number of these will be constant in a given band, but the Gaussian parameters
        vary with source and exposure.

        We'll just track the total count across radius bins; the individual Gaussians
        will know which bin they belong to.

        Fills in the following arrays:
        - self.n_psf_per_source
        - self.psfgauss
        - self.psfgauss_start
        '''

        if not dtype:
            dtype = self.meta_dtype

        self.n_psf_per_source = np.empty(self.n_bands, dtype=dtype)
        self.n_psf_per_source[:] = miniscene.n_psf_per_source  # ???

        psf_dtype = np.dtype([('gauss_params',dtype,6),('sersic_bin',np.int32)])
        n_psfgauss = n_psf_per_source.sum()*self.n_exp*self.n_sources
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)

        self.psfgauss_start = np.zeros(self.n_exp, dtype=np.int32)
        self.psfgauss_start[1:] = np.cumsum(n_psf_per_source)[:-1]*self.n_exp*self.n_sources

        s = 0
        for e in range(self.n_exp):
            for source in miniscene.sources:
                # sources have one set of psf gaussians per exposure
                N = len(source.psfgauss[e])
                self.psfgauss[s:s+N] = source.psfgauss[e]
                s += N
            assert N == psfgauss_start[e]  # check we got our indexing right


    def pack_source_metadata(self, miniscene, dtype=None):
        '''
        We don't actually pack sources in the Patch; that happens
        in a Proposal.  But we do have some global constants related
        to sources, such as the total number of soures and number of
        Sersic radii bins.  So we pack those here.

        Fills in:
        - self.n_sources
        - self.n_radii
        - self.rad2
        '''

        if not dtype:
            dtype = self.meta_dtype

        # number of sources in the patch
        self.n_sources = miniscene.n_active

        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = miniscene.n_radii

        self.rad2 = np.empty(self.n_radii, dtype=dtype)
        self.rad2[:] = miniscene.sources[0].radii**2


    def pack_astrometry(self, sources, dtype=None):
        '''
        The sources know their local astrometric transformation matrices
        (this was previously populated from the stamps).  We need to send
        this data to the GPU so it can apply the sky-to-pixel transformations
        to compare with the image.

        Fills in the following arrays:
        - self.D
        - self.CW
        - self.crpix
        - self.crval
        - self.G
        '''

        if not dtype:
            dtype = self.meta_dtype

        # Each source need different astrometry for each exposure

        self.D = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.CW = np.empty((self.n_exp, self.n_sources, 2, 2), dtype=dtype)
        self.crpix = np.empty((self.n_exp, 2), dtype=dtype)
        self.crval = np.empty((self.n_exp, 2), dtype=dtype)
        self.G = np.empty((self.n_exp), dtype=dtype)

        # The ordering of the astrometric information in the sources is guaranteed
        # to be in the same order as our exposures

        for i in range(self.n_exp):
            # these values are per-exposure
            self.crpix[i] = sources[0].stamp_crpixs[i]
            self.crval[i] = sources[0].stamp_crvals[i]
            self.G[i] = sources[0].stamp_zps[i]

            for s,source in enumerate(sources):
                self.D[i,s] = source.stamp_scales[i]
                self.CW[i,s] = source.stamp_cds[i] # dpix/dra, dpix/ddec



    def pack_pix(self, stamps, mask, super_pixel_size, dtype=None):
        '''
        We have stamps of exposures that we want to pack into
        concatenated 1D pixel arrays.  We want them to be in
        super-pixel order, too.

        As we go, we want to build up the index arrays that
        allow us to find an exposure in the 1D arrays.

        Fills the following arrays:
        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start
        - self.band_N
        - self.exposure_start
        - self.exposure_N
        '''

        # TODO: use mask

        if not dtype:
            dtype = self.pix_dtype
        
        # TODO: convert function to numba

        assert super_pixel_size == 1  # TODO: super-pixel ordering

        shapes = np.array([stamp.shape for stamp in stamps], dtype=int)
        sizes = np.array([stamp.size for stamp in stamps], dtype=int)

        total_padded_size = np.sum(sizes)  # will have super-pixel padding

        self.xpix = np.empty(total_padded_size, dtype=dtype)
        self.ypix = np.empty(total_padded_size, dtype=dtype)
        self.data = np.empty(total_padded_size, dtype=dtype)  # value (i.e. flux) in pixel.
        self.ierr = np.empty(total_padded_size, dtype=dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.empty(self.n_bands, dtype=np.int16)
        self.band_N = np.zeros(self.n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(self.n_exp, dtype=np.int32)
        self.exposure_N = np.empty(self.n_exp, dtype=np.int32)

        i,b = 0,0
        for e,stamp in enumerate(stamps):
            # are we starting a new band?
            if e > 0 and stamp.band != stamp.band[e-1]:
                b += 1
            self.band_N[b] += 1

            N = stamp.size
            self.exposure_start[e] = i
            self.exposure_N[e] = N

            self.xpix[i:i+N] = stamp.xpix
            self.ypix[i:i+N] = stamp.ypix
            self.data[i:i+N] = stamp.vpix
            self.ierr[i:i+N] = stamp.ierr

            # Finished this exposure
            i += N

        assert i == total_padded_size

        self.band_start[0] = 0
        self.band_start[1:] = np.cumsum(self.band_N)[:-1]


    def send_to_gpu(self, residual=False):
        '''
        Transfer the patch data to GPU main memory.  Saves the pointers
        and builds the Patch struct from patch.cu; sends that to GPU memory.
        Saves the struct pointer for forwarding to the likelihood call.

        Parameters
        ----------
        residual: bool, optional
            Whether to allocate GPU-side space for a residual image array.
            Default: False.

        Returns
        -------
        gpu_patch: pycuda.driver.DeviceAllocation
            A device-side pointer to the Patch struct on the GPU.

        '''

        # use names of struct dtype fields to transfer all arrays to the GPU
        self.cuda_ptrs = {}
        for arrname in self.patch_struct_dtype.names:
            try:
                arr = getattr(self, arrname)
                arr_dt = self.patch_struct_dtype[arrname]
                if arr_dt == self.ptr_dtype:
                    print(f'Copying {arrname} to GPU: {arr} (dtype: {arr.dtype})')
                    self.cuda_ptrs[arrname] = cuda.to_device(arr)
                    print(f'self.cuda_ptrs[arrname] = {self.cuda_ptrs[arrname]}')
                    print(f'hex(int(self.cuda_ptrs[arrname])) = {hex(int(self.cuda_ptrs[arrname]))}')
            except AttributeError:
                pass  # this will be removed later once we have all attrs


        # use their pointers to fill in the patch struct
        #assert set(self.cuda_ptrs) == set(self.patch_struct_dtype.names)

        # Get a singlet of the custom dtype
        # Is there a better way to do this?
        patch_struct = np.empty(1, dtype=self.patch_struct_dtype)[0]

        for arrname in self.patch_struct_dtype.names:
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                if arrname not in self.cuda_ptrs:
                    continue
                print('Assigning pointer for',arrname)
                # Array? Assign pointer.
                patch_struct[arrname] = self.cuda_ptrs[arrname]
            else:
                # Value? Assign directly.
                patch_struct[arrname] = getattr(self, arrname)

        # Copy the patch struct to the gpu
        print(patch_struct)
        self.gpu_patch = cuda.to_device(patch_struct)

        return self.gpu_patch

    def free(self):
        # Release the device-side arrays
        try:
            for cuda_ptr in self.cuda_ptrs.values():
                if cuda_ptr:
                    cuda_ptr.free()
        except AttributeError:
            pass  # no cuda_ptrs

        # Release the device-side patch struct
        try:
            if self.gpu_patch:
                self.gpu_patch.free()
        except AttributeError:
            pass  # no gpu_patch


    def __del__(self):
        self.free()  # do we want to do this?


    def test_struct_transfer(self, gpu_patch):
        '''
        Run a simple PyCUDA kernel that checks that the data sent
        was the data received.
        '''

        from pycuda.compiler import SourceModule
        import os

        mod = SourceModule(
            f'''
            #include <limits.h>
            #include "patch.cu"

            __global__ void check_patch_struct(Patch *patch){{
                printf("Kernel sees: sizeof(Patch) = %d (Numpy size: {self.patch_struct_dtype.itemsize})\\n", sizeof(Patch));
                assert(sizeof(Patch) == {self.patch_struct_dtype.itemsize});
                
                printf("Kernel sees: patch->n_sources = %d\\n", patch->n_sources);
                assert(patch->n_sources == {self.n_sources});

                printf("Kernel sees: patch->n_radii = %d\\n", patch->n_radii);
                printf("Kernel sees: patch->rad2 = %p\\n", patch->rad2);
                for(int i = 0; i < patch->n_radii; i++){{
                    printf("%f ", patch->rad2[i]);
                }}
                printf("\\n");
            }}
            ''',
            include_dirs=[os.environ['HOME'] + '/forcepho/forcepho'], cache_dir='/gpfs/wolf/gen126/scratch/lgarrison/pycuda_cache')

        kernel = mod.get_function('check_patch_struct')

        retcode = kernel(self.gpu_patch, block=(1,1,1), grid=(1,1,1))

        print(f"Kernel done.")


    # TODO: will probably go elsewhere
    def send_proposal(self, proposal, block=1024):
        grid = (self.n_bands,1,1)
        block = (block,1,1)

        kernel(self.gpu_patch, proposal, results, grid=grid, block=block)


    # The following must be kept bitwise identical to patch.cu!
    # TODO: compare the dtype size to the struct size
    ptr_dtype = np.uintp  # pointer type
    patch_struct_dtype = np.dtype([('data',ptr_dtype),
                                   ('ierr',ptr_dtype),
                                   ('xpix',ptr_dtype),
                                   ('ypix',ptr_dtype),
                                   ('residual',ptr_dtype),

                                   ('exposure_start',ptr_dtype),
                                   ('exposure_N',ptr_dtype),

                                   ('band_start',ptr_dtype),
                                   ('band_N',ptr_dtype),

                                   ('n_sources', np.int32),
                                   ('n_radii', np.int32),

                                   ('rad2',ptr_dtype),

                                   ('D',ptr_dtype),

                                   ('crpix',ptr_dtype),
                                   ('crval',ptr_dtype),
                                   ('CW',ptr_dtype),
                                   ('G',ptr_dtype),

                                   ('n_psf_per_source',ptr_dtype),

                                   ('psfgauss',ptr_dtype),
                                   ('psfgauss_start',ptr_dtype),

                                   ], align=True)
