#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""patch.py

Data model for a patch on the sky. The corresponding GPU-side CUDA struct is
in patch.cu.
"""

import numpy as np

from .. import source_dir

try:
    import pycuda
    import pycuda.driver as cuda
except ImportError:
    pass

__all__ = ["Patch"]


class Patch:

    """Base class for objects that pack pixel and scene metadata into
    structures in a manner suitable for sending to the GPU. This includes
    rearranging the data into (padded) super-pixel order.

    The Patch object contains methods for sending the patch data to the GPU
    with PyCUDA.

    Parameters
    ----------
    return_residual: bool, optional
        Whether the residual image will be returned from the GPU.
        Default: False.

    pix_dtype: np.dtype, optional
        The Numpy datatype of the pixel data, like fluxes and coordinates.
        Default: np.float32

    meta_dtype: np.dtype, optional
        The Numpy datatype of the non-pixel data, like astrometry.
        Default: np.float32
    """

    # The following must be kept bitwise identical to patch.cu!
    # TODO: compare the dtype size to the struct size
    ptr_dtype = np.uintp  # pointer type
    pdt = np.dtype([('data', ptr_dtype),
                    ('ierr', ptr_dtype),
                    ('xpix', ptr_dtype),
                    ('ypix', ptr_dtype),
                    ('residual', ptr_dtype),

                    ('exposure_start', ptr_dtype),
                    ('exposure_N', ptr_dtype),

                    ('band_start', ptr_dtype),
                    ('band_N', ptr_dtype),

                    ('n_sources', np.int32),
                    ('n_radii', np.int32),

                    ('rad2', ptr_dtype),

                    ('D', ptr_dtype),
                    ('crpix', ptr_dtype),
                    ('crval', ptr_dtype),
                    ('CW', ptr_dtype),

                    ('G', ptr_dtype),

                    ('n_psf_per_source', ptr_dtype),
                    ('psfgauss', ptr_dtype),
                    ('psfgauss_start', ptr_dtype),

                    ], align=True)

    patch_struct_dtype = pdt
    meta_names = ["n_sources", "n_radii", "rad2",
                  "D", "crpix", "crval", "CW", "G",
                  "n_psf_per_source", "psfgauss", "psfgauss_start"]

    def __init__(self,
                 return_residual=False,
                 pix_dtype=np.float32,   # data type for pixel and flux data
                 meta_dtype=np.float32,  # data type for non-pixel data
                 ):
        self.pix_dtype = pix_dtype
        self.meta_dtype = meta_dtype
        self.return_residual = return_residual

    @property
    def npix(self):
        try:
            n = self.xpix.shape[0]
            return n
        except(AttributeError):
            return 0

    def build_patch(self, npix=1, n_sources=1, n_radii=1,
                    n_bands=1, n_exp=1):
        """This is an abstract method that takes scene and exposure data and
        packs it all into arrays with the proper shapes and data types for
        transfer to GPU.

        This should be overridden by  subclasses, which might get the data in
        different formats.

        Fills in the following arrays:

        - self.xpix
        - self.ypix
        - self.data
        - self.ierr
        - self.band_start     [NBAND] exposure index corresponding to the start of each band
        - self.band_N         [NBAND] number of exposures in each band
        - self.exposure_start [NEXP]  pixel index corresponding to the start of each exposure
        - self.exposure_N     [NEXP]  number of pixels (including warp padding) in each exposure

        - self.n_sources     [1]
        - self.n_radii       [1]
        - self.rad2          [N_RADII]

        - self.D             [NEXP, NSOURCE, 2, 2]
        - self.CW            [NEXP, NSOURCE, 2, 2]
        - self.crpix         [NEXP, NSOURCE, 2]
        - self.crval         [NEXP, NSOURCE, 2]
        - self.G             [NEXP]

        - self.n_psf_per_source  [NBAND]  number of PSFGaussians per source in each band
        - self.psfgauss          [NPSFG]  An array of PSFGaussian parameters
        - self.psfgauss_start    [NEXP]   PSFGaussian index corresponding to the start of each exposure.
        """
        self.n_exp = n_exp

        # --- Pixel Data ---
        # These are arrays of pixel data for *all* pixels in a patch (i.e.
        # multiple exposures)
        self.xpix = np.zeros(npix, dtype=self.pix_dtype)
        self.ypix = np.zeros(npix, dtype=self.pix_dtype)
        self.data = np.zeros(npix, dtype=self.pix_dtype)  # flux
        self.ierr = np.zeros(npix, dtype=self.pix_dtype)  # 1/sigma

        # These index the exposure_start and exposure_N arrays
        # bands are indexed sequentially, not by any band ID
        self.band_start = np.empty(n_bands, dtype=np.int16)
        self.band_N = np.zeros(n_bands, dtype=np.int16)

        # These index the pixel arrays (also sequential)
        self.exposure_start = np.empty(n_exp, dtype=np.int32)
        self.exposure_N = np.empty(n_exp, dtype=np.int32)

        # --- Sources ---
        self.n_sources = n_sources
        # number of gaussians in the sersic
        # should be the same for all sources
        self.n_radii = n_radii
        self.rad2 = np.empty(self.n_radii, dtype=self.meta_dtype)

        # --- Meta Data ---
        # Each source need different astrometry for each exposure
        self.D = np.empty((n_exp, self.n_sources, 2, 2), dtype=self.meta_dtype)
        self.CW = np.empty((n_exp, self.n_sources, 2, 2), dtype=self.meta_dtype)
        self.crpix = np.empty((n_exp, self.n_sources, 2), dtype=self.meta_dtype)
        self.crval = np.empty((n_exp, self.n_sources, 2), dtype=self.meta_dtype)
        self.G = np.empty((n_exp), dtype=self.meta_dtype)

        # --- PSF data ---
        # Each source gets a different set of PSF gaussians for each exposure.
        # But, the number of Gaussians per source is constant within a band.
        psf_dtype = np.dtype([('gauss_params', self.meta_dtype, 6),
                              ('sersic_bin', np.int32)])
        self.n_psf_per_source = np.empty(n_bands, dtype=np.int32)
        n_psfgauss = self.n_psf_per_source * self.band_N * n_sources
        self.psfgauss = np.empty(n_psfgauss, dtype=psf_dtype)
        self.psfgauss_start = np.zeros(n_exp, dtype=np.int32)

    def send_to_gpu(self):
        """Transfer all the patch data to GPU main memory.  Saves the pointers
        and builds the Patch struct from patch.cu; sends that to GPU memory.
        Saves the struct pointer for forwarding to the likelihood call.

        Parameters
        ----------

        Returns
        -------
        gpu_patch: pycuda.driver.DeviceAllocation
            A device-side pointer to the Patch struct on the GPU.
        """
        # use names of struct dtype fields to transfer all arrays to the GPU
        self.cuda_ptrs = {}
        for arrname in self.patch_struct_dtype.names:
            try:
                arr = getattr(self, arrname)
                arr_dt = self.patch_struct_dtype[arrname]
                if arr_dt == self.ptr_dtype:
                    self.cuda_ptrs[arrname] = cuda.to_device(arr)
            except AttributeError:
                if arrname != 'residual':
                    raise

        if self.return_residual:
            self.cuda_ptrs['residual'] = cuda.mem_alloc(self.xpix.nbytes)

        _ = self.send_patchstruct_to_gpu()

        return self.gpu_patch

    def swap_on_gpu(self):
        """This method does several things:
            1) Free existing meta-data arrays on the device, and send new
               (assumed already packed) metadata arrays to device,
               replacing the associated CUDA pointers;
            2) Swap the CUDA pointers for the data and the residual;
            3) Free the existing device-side patch_struct;
            4) Refill the patch_struct array of CUDA pointers and values,
               and send to device

        After this call the GPU uses the former "residual" vector as the "data"
        """
        assert "residual" in self.cuda_ptrs
        # Replace the metadata on the GPU, as well as the Cuda pointers
        # This releases the device side arrays corresponding to old metadata
        for arrname in self.meta_names:
            try:
                arr = getattr(self, arrname)
                arr_dt = self.patch_struct_dtype[arrname]
                if arr_dt == self.ptr_dtype:
                    self.cuda_ptrs[arrname].free()
                    self.cuda_ptrs[arrname] = cuda.to_device(arr)
            except AttributeError:
                if arrname != 'residual':
                    raise
        # Swap Cuda pointers for data and residual
        self.cuda_ptrs["data"], self.cuda_ptrs["residual"] = self.cuda_ptrs["residual"], self.cuda_ptrs["data"]
        # Pack pointers into structure and send structure to device
        _ = self.send_patchstruct_to_gpu()
        return self.gpu_patch

    def send_patchstruct_to_gpu(self):
        """Create new patch_struct and fill with values and with CUDA pointers
        to GPU arrays, and send the patch_struct to the GPU.
        """
        # Get a singlet of the custom dtype
        # Is there a better way to do this?
        patch_struct = np.zeros(1, dtype=self.patch_struct_dtype)[0]

        for arrname in self.patch_struct_dtype.names:
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                if arrname not in self.cuda_ptrs:
                    continue
                # Array? Assign pointer.
                patch_struct[arrname] = self.cuda_ptrs[arrname]
            else:
                # Value? Assign directly.
                patch_struct[arrname] = getattr(self, arrname)

        # Release the device-side patch struct, if it exists
        # Do this before freeing and replacing meta-data arrays on device?
        try:
            if self.gpu_patch:
                self.gpu_patch.free()
                del self.gpu_patch
        except(AttributeError):
            pass  # no gpu_patch

        # Copy the new patch struct to the gpu
        self.gpu_patch = cuda.to_device(patch_struct)

        return self.gpu_patch

    def free(self):
        # Release ALL the device-side arrays
        try:
            for cuda_ptr in self.cuda_ptrs.values():
                if cuda_ptr:
                    cuda_ptr.free()
            self.cuda_ptrs = {}
        except(AttributeError):
            pass  # no cuda_ptrs

        # Release the device-side patch struct
        try:
            if self.gpu_patch:
                self.gpu_patch.free()
                del self.gpu_patch
        except(AttributeError):
            pass  # no gpu_patch

    def __del__(self):
        self.free()  # do we want to do this?

    def test_struct_transfer(self, gpu_patch, cache_dir=False):
        """Run a simple PyCUDA kernel that checks that the data sent was the
        data received.
        """

        from pycuda.compiler import SourceModule
        import os
        thisdir = os.path.abspath(os.path.dirname(__file__))

        mod = SourceModule(
            """
            #include <limits.h>
            #include "patch.cu"

            __global__ void check_patch_struct(Patch *patch){{
                printf("Kernel sees: sizeof(Patch) = %d (Numpy size: {sz})\\n", sizeof(Patch));
                assert(sizeof(Patch) == {sz});

                printf("Kernel sees: patch->n_sources = %d\\n", patch->n_sources);
                assert(patch->n_sources == {nsource});

                printf("Kernel sees: patch->n_radii = %d\\n", patch->n_radii);
                printf("Kernel sees: patch->rad2 = %p\\n", patch->rad2);
                for(int i = 0; i < patch->n_radii; i++){{
                    printf("%g ", patch->rad2[i]);
                }}
                printf("\\n");

                int i = 100;
                PSFSourceGaussian p = patch->psfgauss[i];
                printf("Kernel sees: patch->psfgauss[%d] = (%f,%f,%f,%f,%f,%f,%d)\\n", i,
                        p.amp, p.xcen, p.ycen,
                        p.Cxx, p.Cyy, p.Cxy, p.sersic_radius_bin
                        );
            }}
            """.format(sz=self.patch_struct_dtype.itemsize, nsource=self.n_sources),
            include_dirs=[source_dir, thisdir],
            cache_dir=cache_dir)

        print(self.psfgauss[100])
        kernel = mod.get_function('check_patch_struct')
        retcode = kernel(gpu_patch, block=(1, 1, 1), grid=(1, 1, 1))
        print("Kernel done.")
