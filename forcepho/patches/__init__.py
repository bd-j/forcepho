from .patch import Patch
from .device_patch import GPUPatchMixin, CPUPatchMixin
from .pixel_patch import PixelPatch, StorePatch, FITSPatch

__all__ = ["Patch",
           "PixelPatch",
           "StorePatch", "FITSPatch",
           "JadesPatch",
           "SimplePatch"]


class JadesPatch(StorePatch, GPUPatchMixin):
    pass


class SimplePatch(FITSPatch, GPUPatchMixin):
    pass
