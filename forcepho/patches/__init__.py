from .patch import Patch
from .device_patch import GPUPatchMixin, CPUPatchMixin
from .pixel_patch import PixelPatch, StorePatch, FITSPatch, DirectPatch

__all__ = ["Patch",
           "PixelPatch",
           "StorePatch", "FITSPatch", "DirectPatch",
           "JadesPatch",
           "SimplePatch"]


class JadesPatch(StorePatch, GPUPatchMixin):
    pass


class SimplePatch(FITSPatch, GPUPatchMixin):
    pass
