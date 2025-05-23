# -*- coding: utf-8 -*-

"""device_patch.py - mix-in classes containing methods for communicating with GPU or CPU device
"""

import numpy as np

from ..sources import Scene, Galaxy
from ..proposal import GPUProposer, CPUProposer
from ..model import FastPosterior, Transform
from ..superscene import bounds_vectors

from .. import source_dir
try:
    import pycuda
    import pycuda.driver as cuda
except (ImportError, RuntimeError):
    pass

__all__ = ["GPUPatchMixin", "CPUPatchMixin"]


class DevicePatchMixin:

    """Abstract Base Class for device communication of Patch data.
    """

    def prepare_model(self, active=None, fixed=None, big=None,
                      bounds=None, maxactive=15, shapes=Galaxy.SHAPE_COLS,
                      big_scene_kwargs={}, model_kwargs={}):
        """Prepare the patch for sampling/evaluation.  This includes subtracting
        big and fixed sources, and wrapping the patch in a FastPosterior
        including transforms.

        Parameters
        ----------
        active : structured ndarray of shape (n_active_sources,)
            Catalog of active source parameters

        fixed : structured ndarray of shape (n_fixed_sources,)
            Catalog of fixed source parameters

        maxactive : int
            Maximum number of active sources at one time.

        Returns
        -------
        model : instance of forcepho.model.GPUPosterior
            The model object for this patch

        q : ndarray of shape (n_dim,)
            The initial parameter vector.
        """
        if shapes is None:
            shapes = Galaxy.SHAPE_COLS

        proposer, scene = self.subtract_fixed(fixed, active=active, big=big,
                                              big_scene_kwargs=big_scene_kwargs,
                                              maxactive=maxactive)
        self.return_residual = False
        q = scene.get_all_source_params().copy()

        if bounds is None:
            model = FastPosterior(proposer, scene=scene, patch=self,
                                  transform=Transform(len(q)), **model_kwargs)
        else:
            lo, hi = bounds_vectors(bounds, self.bandlist, shapenames=shapes,
                                    reference_coordinates=self.patch_reference_coordinates)
            model = FastPosterior(proposer, scene=scene, patch=self,
                                  lower=lo, upper=hi, **model_kwargs)

        return model, q

    def send_to_device(self):
        """Transfer all the patch data arrays to the device, collecting a
        dicionary of pointers in the process. creates a dictionary of device
        side pointers to the patch data, sending the individual arrays in the
        process if necessary.
        """
        raise NotImplementedError

    def swap_on_device(self):
        """This method does several things:
        1) Free existing meta-data arrays on the device, and send new
            (assumed already packed) metadata arrays to device,
            replacing the associated pointers;
        2) Swap the device-side pointers for the data and the residual;
        3) Free the existing device-side patch_struct;
        4) Refill the patch_struct array of pointers and values,
            and send to device

        After this call the device uses the former "residual" vector as the
        "data".
        """
        raise NotImplementedError

    def replace_device_meta_ptrs(self):
        raise NotImplementedError

    def send_patchstruct_to_device(self):
        """Create new patch_struct and fill with values or pointers to arrays
        and for the GPU send the patch_struct to the GPU.  Return a pointer to
        the device side patchstruct.
        """
        raise NotImplementedError

    def retrieve_array(self, **kwargs):
        raise NotImplementedError

    def get_proposer(self):
        raise NotImplementedError

    def subtract_fixed(self, fixed, active=None, big=None, maxactive=15,
                       swap_local=False, big_scene_kwargs={}, **scene_kwargs):
        """Subtract a set of big and/or fixed sources from the data on the GPU.

        Leaves the GPU-side *data* array filled with "residuals" from
        subtracting the fixed sources, and either the last set of fixed sources
        or the active sources (if supplied) in the GPU-side meta data arrays.
        Optionally also fill the CPU-side data array with these residuals.

        Parameters
        ----------
        fixed : structured array of shape (n_fixed_sources,)
            Catalog of fixed source parameters for sources to subtract from the
            image.

        active : structured array of shape (n_fixed_sources,)
            Catalog of active source parameters.  These wil be sent to the

        Returns
        -------
        proposer : instance of forcepho.proposal.Proposer
            Communicator to the GPU for proposals

        scene : instance of forcepho.sources.Scene
            Scene corresponding to the meta-data currently on the GPU.  If
            `active` was supplied then this will be the Scene for the active
            sources
        """
        # Build all the scenes we want to evaluate and subtract. The last scene
        # will not actually be subtracted, but will have meta data transferred
        # to GPU and ready to accept proposals.
        scenes = []
        if big is not None:
            inds = np.arange(maxactive, len(big), maxactive)
            blocks = np.array_split(big, inds)
            scenes.extend([self.set_scene(block, **big_scene_kwargs) for block in blocks])
        if fixed is not None:
            inds = np.arange(maxactive, len(fixed), maxactive)
            blocks = np.array_split(fixed, inds)
            scenes.extend([self.set_scene(block, **scene_kwargs) for block in blocks])
        if active is not None:
            scenes.append(self.set_scene(active, **scene_kwargs))
        else:
            print("Warning: The meta-data will be for a scene already subtracted from the GPU-side data array!")
            assert swap_local
            scenes.append(scenes[-1])

        assert len(scenes) > 0, "No scenes to prepare!"

        # to communicate with the device
        proposer = self.get_proposer()

        # Pack first scene and send it with pixel data
        self.return_residual = True
        assert not self._dirty_data
        self.pack_meta(scenes[0])
        _ = self.send_to_device()
        # loop over scenes
        for i, scene in enumerate(scenes[1:]):
            # evaluate previous scene
            proposal = scenes[i].get_proposal()
            out = proposer.evaluate_proposal(proposal, patch=self)
            # pack this scene
            self.pack_meta(scene)
            # replace on-device data with on-device residual
            # replace meta of previous block with this block
            # on-device residual is now garbage
            # original data is gone!
            self.swap_on_device()

        return proposer, scenes[-1]

    def design_matrix(self, active=None, proposer=None, **scene_kwargs):
        """Assumes :py:meth:`subtract_fixed` was already run.

        Parameters
        ----------
        active : structured array
            Catalog of sources to get the models for

        Extra Parameters
        ----------------
        scene_kwargs : dictionary
            Keyword arguments for JadesPatch.set_scene()

        Returns
        -------
        Xes : list of ndarrays, each of shape (n_source, n_pix_band)
            The design matrix for each band, i.e. X s.t.
               :math:`Y[band] = f[band] \dot Xes[band]`
            where f[band] is a vector of source fluxes in 'band'

        ys : list of ndarrays of shape (n_bix_band,)
           The target data, after all fixed and big sources subtracted
        """
        # Get the residuals after subtracting anything fixed
        y = self.retrieve_array("data")
        ys = np.split(y, np.cumsum(self.band_N_pix[:-1]))

        Xes = [np.zeros((len(active), n)) for n in self.band_N_pix]
        scenes = [self.set_scene([a], **scene_kwargs)
                  for a in active]

        # to communicate with the GPU
        if not proposer:
            proposer = self.get_proposer()

        # Pack first scene and send it with pixel data
        self.return_residual = True
        # loop over scenes/sources
        for i, scene in enumerate(scenes):
            # get the metadata for this scene packed and sent (data pointers do not change)
            self.pack_meta(scene)
            self.replace_device_meta_ptrs()
            self.send_patchstruct_to_device()

            # evaluate scene
            proposal = scene.get_proposal()
            _, _, residual = proposer.evaluate_proposal(proposal, patch=self, unpack=False)
            # get model and split by band, normalize
            model = y - residual
            split = np.split(model, np.cumsum(self.band_N_pix[:-1]))
            for j, b in enumerate(self.bandlist):
                Xes[j][i, :] = split[j] / active[i][b]

        return Xes, ys

    def render(self, catalog, splinedata=None):
        """Convenience method that allows one to render a particular scene
        represented as rows of a structured array, and return images
        """
        proposer = self.get_proposer()
        scene = self.set_scene(catalog, splinedata=splinedata)
        self.pack_meta(scene)

        proposal = scene.get_proposal()
        orig, self.return_residual = self.return_residual, True
        ret = proposer.evaluate_proposal(proposal, patch=self, unpack=True)
        self.return_residual = True
        chi2, chi2_derivs, residuals = ret
        image = -residuals
        return image

    def free(self):
        pass


class GPUPatchMixin(DevicePatchMixin):

    """Mix-in class for communicating patch data with the GPU using PyCUDA.
    """

    def get_proposer(self):
        return GPUProposer()

    def send_to_device(self):
        """Transfer all the patch data to GPU main memory.  Saves the pointers
        and builds the Patch struct from patch.cc; sends that to GPU memory.
        Saves the struct pointer for forwarding to the likelihood call.

        Parameters
        ----------

        Returns
        -------
        device_patch: pycuda.driver.DeviceAllocation
            A host-side pointer to the Patch struct on the GPU device.
        """
        # use names of struct dtype fields to transfer all arrays to the GPU
        self.device_ptrs = {}
        for arrname in self.patch_struct_dtype.names:
            try:
                arr = getattr(self, arrname)
                arr_dt = self.patch_struct_dtype[arrname]
                if arr_dt == self.ptr_dtype:
                    self.device_ptrs[arrname] = cuda.to_device(arr)
            except AttributeError:
                if arrname != 'residual':
                    raise

        if self.return_residual:
            self.device_ptrs['residual'] = cuda.mem_alloc(self.xpix.nbytes)

        _ = self.send_patchstruct_to_device()

        self._dirty_data = False

        return self.device_patch

    def replace_device_meta_ptrs(self):
        """Replace the metadata on the GPU, as well as the Cuda pointers. This
        also releases the device side arrays corresponding to old metadata.
        """
        for arrname in self.meta_names:
            arr = getattr(self, arrname)
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                self.device_ptrs[arrname].free()
                self.device_ptrs[arrname] = cuda.to_device(arr)

    def send_patchstruct_to_device(self):
        """Create new patch_struct and fill with values and with CUDA pointers
        to GPU arrays, and send the patch_struct to the GPU.
        """
        # Get a singlet of the custom dtype
        # Is there a better way to do this?
        patch_struct = np.zeros(1, dtype=self.patch_struct_dtype)[0]

        for arrname in self.patch_struct_dtype.names:
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                if arrname not in self.device_ptrs:
                    continue
                # Array? Assign pointer.
                patch_struct[arrname] = self.device_ptrs[arrname]
            else:
                # Value? Assign directly.
                patch_struct[arrname] = getattr(self, arrname)

        # Release the device-side patch struct, if it exists
        # Do this before freeing and replacing meta-data arrays on device?
        try:
            if self.device_patch:
                self.device_patch.free()
                del self.device_patch
        except(AttributeError):
            pass  # no gpu_patch

        # Copy the new patch struct to the gpu
        self.patch_struct = patch_struct
        self.device_patch = cuda.to_device(patch_struct)
        return self.device_patch

    def swap_on_device(self):
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
        assert "residual" in self.device_ptrs, "Must instantiate the Patch with `return_residual=True`"
        # replace gpu meta data pointers with currently packed meta
        self.replace_device_meta_ptrs()
        # Swap Cuda pointers for data and residual
        self.device_ptrs["data"], self.device_ptrs["residual"] = self.device_ptrs["residual"], self.device_ptrs["data"]
        # Pack pointers into structure and send structure to device
        _ = self.send_patchstruct_to_device()
        self._dirty_data = True

        return self.device_patch

    def retrieve_array(self, arrname="residual"):
        """Retrieve a pixel array from the GPU
        """
        flatdata = cuda.from_device(self.device_ptrs[arrname],
                                    shape=self.xpix.shape,
                                    dtype=self.pix_dtype)
        return flatdata

    def free(self):
        # Release ALL the device-side arrays
        try:
            for cuda_ptr in self.device_ptrs.values():
                if cuda_ptr:
                    cuda_ptr.free()
            self.device_ptrs = {}
        except(AttributeError):
            pass  # no cuda_ptrs

        # Release the device-side patch struct
        try:
            if self.device_patch:
                self.device_patch.free()
                del self.device_patch
        except(AttributeError):
            pass  # no gpu_patch

    def test_struct_transfer(self, gpu_patch, cache_dir=False, psf_index=35):
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

                int i = {ip};
                PSFSourceGaussian p = patch->psfgauss[i];
                printf("Kernel sees: patch->psfgauss[%d] = (%f,%f,%f,%f,%f,%f,%d)\\n", i,
                        p.amp, p.xcen, p.ycen,
                        p.Cxx, p.Cyy, p.Cxy, p.sersic_radius_bin
                        );
            }}
            """.format(sz=self.patch_struct_dtype.itemsize, nsource=self.n_sources,
                       ip=psf_index),
            include_dirs=[source_dir, thisdir],
            cache_dir=cache_dir)

        print(self.psfgauss[psf_index])
        kernel = mod.get_function('check_patch_struct')
        retcode = kernel(gpu_patch, block=(1, 1, 1), grid=(1, 1, 1))
        print("Kernel done.")


class CPUPatchMixin(DevicePatchMixin):

    """Mix-in class for communicating patch data with the CPU
    """

    def get_proposer(self):
        return CPUProposer()

    def send_to_device(self):
        """Transfer pointers to all patch data.  Saves the pointers
        and builds the Patch struct from patch.cc; . Saves the struct pointer
        for forwarding to the likelihood call.

        Parameters
        ----------

        Returns
        -------
        device_patch: self.ptr_dtype
            A host-side pointer to the Patch struct (which is itself filled with
            pointers)
        """
        self.device_ptrs = {}
        self.buffer_sizes = {}
        # TODO: is this the right place for this?
        self.residual = np.zeros_like(self.data)

        # Create dictionary 'device_ptrs' of pointers to relevant arrays
        for arrname in self.patch_struct_dtype.names:
            try:
                arr = getattr(self, arrname)
                arr_dt = self.patch_struct_dtype[arrname]
                if arr_dt == self.ptr_dtype:
                    ptr, rof = arr.__array_interface__['data']
                    self.device_ptrs[arrname] = ptr
            except AttributeError:
                if arrname != 'residual':
                    raise
        # fill 'patchstruct' with the pointers, get a pointer to the patchstruct
        self.device_patch = self.send_patchstruct_to_device()
        return self.device_patch

    def replace_device_meta_ptrs(self):
        """Replace the metadata on the GPU, as well as the Cuda pointers. This
        also releases the device side arrays corresponding to old metadata.
        """
        for arrname in self.meta_names:
            arr = getattr(self, arrname)
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                ptr, rof = arr.__array_interface__['data']
                # TODO: should we free data asociated with the existing ptr?
                self.device_ptrs[arrname] = ptr

    def send_patchstruct_to_device(self):
        """Create a patchstruct array and fill it with pointers.

        Returns
        -------
        patchstruct_ptr : ptr_dtype
            A pointer to the patchstruct array of pointers.
        """
        # Get a singlet of the custom dtype
        patch_struct = np.zeros(1, dtype=self.patch_struct_dtype)[0]

        for arrname in self.patch_struct_dtype.names:
            arr_dt = self.patch_struct_dtype[arrname]
            if arr_dt == self.ptr_dtype:
                # Array?  assign pointer
                patch_struct[arrname] = self.device_ptrs[arrname]
            else:
                # Value? Assign directly
                patch_struct[arrname] = getattr(self, arrname)

        # get a ptr to the patchstruct (of pointers)
        self.patch_struct = patch_struct
        self.patchstruct_ptr, rof = patch_struct.__array_interface__['data']
        self.device_patch = self.ptr_dtype(self.patchstruct_ptr)
        return self.device_patch

    def swap_on_device(self):
        """This method does several things:
        1) Free existing meta-data arrays on the device, and send new
            (assumed already packed) metadata arrays to device,
            replacing the associated pointers;
        2) Swap the device-side pointers for the data and the residual;
        3) Swap the data and residual attribute references;
        3) Free the existing device-side patch_struct;
        4) Refill the patch_struct array of pointers and values,
            and send to device

        After this call the device uses the former "residual" vector as the "data".
        """
        assert "residual" in self.device_ptrs, "Must instantiate the Patch with `return_residual=True`"
        # replace device meta data pointers with currently packed meta
        self.replace_device_meta_ptrs()
        # Swap pointers for data and residual
        # kernel should now operate on what was once the residual
        self.device_ptrs["data"], self.device_ptrs["residual"] = self.device_ptrs["residual"], self.device_ptrs["data"]
        # swap attribute references too!
        self.data, self.residual = self.residual, self.data
        # Pack pointers into structure and send structure to device
        _ = self.send_patchstruct_to_device()
        #self._dirty_data = True

        return self.device_patch

    def retrieve_array(self, arrname="residual"):
        """Retrieve a copy of pixel array from the CPU
        """
        # we just return a copy of the array
        flatdata = getattr(self, arrname)
        #shape=self.xpix.shape, dtype=self.pix_dtype)

        return flatdata.copy()
