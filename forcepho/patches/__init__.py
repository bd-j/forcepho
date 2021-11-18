from .patch import Patch
from .device_patch import GPUPatchMixin
from .pixel_patch import StorePatch, FITSPatch

__all__ = ["Patch",
           "StorePatch", "FITSPatch",
           "JadesPatch",
           "SimplePatch"]

class JadesPatch(StorePatch, GPUPatchMixin):
    pass


class SimplePatch(FITSPatch, GPUPatchMixin):
    pass
