"""Register/spill count and actual device-memory-per-point, via the CUDA **driver**
API directly (`ctypes` + `libcuda.so.1`) -- `cuobjdump`/`nvdisasm`/`nvcc` are all absent
in this environment, but `cuFuncGetAttribute`/`cuMemGetInfo` need none of them.
"""
from __future__ import annotations

import ctypes
import glob
import re

_cuda = None


def _lib():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuInit(0)
    return _cuda


def kernel_registers(kernel_cache_dir: str, module_hash_prefix: str) -> dict:
    """`{"forward": {"regs": int, "local_bytes": int}, "backward": {...}}` for the
    Warp-generated CUDA kernel whose cached module dir starts with
    `module_hash_prefix` (Warp names it `<file_stem>_<content_hash>`). Returns `{}` if
    no matching `.cubin` is found (e.g. CPU-only, or not compiled for CUDA yet)."""
    cubin_matches = glob.glob(f"{kernel_cache_dir}/*{module_hash_prefix}*/*.cubin")
    if not cubin_matches:
        return {}
    cubin_path = cubin_matches[0]

    cuda = _lib()
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    try:
        mod = ctypes.c_void_p()
        if cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()) != 0:
            return {}

        # Symbol names look like `<kernel>_<8hex>_cuda_kernel_{forward,backward}` --
        # found via `strings` earlier; regex them out of the raw file instead of
        # shelling out again.
        raw = open(cubin_path, "rb").read()
        names = set(re.findall(rb"[\w]+_cuda_kernel_(?:forward|backward)", raw))

        out = {}
        for kname in names:
            func = ctypes.c_void_p()
            if cuda.cuModuleGetFunction(ctypes.byref(func), mod, kname) != 0:
                continue
            regs, local_bytes = ctypes.c_int(), ctypes.c_int()
            cuda.cuFuncGetAttribute(ctypes.byref(regs), 4, func)          # NUM_REGS
            cuda.cuFuncGetAttribute(ctypes.byref(local_bytes), 3, func)   # LOCAL_SIZE_BYTES
            which = "backward" if kname.endswith(b"backward") else "forward"
            out[which] = {"regs": regs.value, "local_bytes": local_bytes.value,
                          "symbol": kname.decode()}
        return out
    finally:
        cuda.cuCtxDestroy_v2(ctx)


def device_mem_used_bytes() -> int:
    """`total - free` for the whole device (not per-context -- `cuMemGetInfo` reports
    device-wide usage regardless of which context asks), in bytes. Reflects whatever
    Warp AND JAX have already allocated on this device, since both share the one GPU
    in this process -- that is deliberate: the question is the device's real headroom,
    not one framework's accounting of it."""
    cuda = _lib()
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    # A context already current (Warp's own) works too; retain-or-create either way.
    cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    created = False
    if not ctx.value:
        cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
        created = True
    try:
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        return total.value - free.value
    finally:
        if created:
            cuda.cuCtxDestroy_v2(ctx)
