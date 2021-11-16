
from .device_patch import GPUPatchMixin
from .pixel_patch import StorePatch, FITSPatch


class JadesPatch(StorePatch, GPUPatchMixin):
    pass


class SimplePatch(FITSPatch, GPUPatchMixin):
    pass


__all__ = ["Patch",
           "JadesPatch",
           "SimplePatch"]