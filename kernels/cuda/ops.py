"""PyTorch-facing wrappers around the CUDA kernels in this directory."""
from __future__ import annotations

import ctypes
from functools import lru_cache

import torch

from ._build import load

VECTOR_ADD_VARIANTS = ("naive", "float4")
REDUCE_VARIANTS = ("naive_atomic", "smem_tree", "smem_tree_gs", "warp_shuffle")
REDUCE_OPS = ("sum", "max")

_p, _i64 = ctypes.c_void_p, ctypes.c_longlong


@lru_cache(maxsize=None)
def _fn(source: str, name: str, nptrs: int):
    lib = load(source)
    fn = getattr(lib, name)
    fn.argtypes = [_p] * nptrs + [_i64, _p]  # pointers..., n, stream
    fn.restype = ctypes.c_int
    lib.cuda_error_string.argtypes = [ctypes.c_int]
    lib.cuda_error_string.restype = ctypes.c_char_p
    return fn, lib


def _launch(source: str, name: str, *tensors: torch.Tensor, n: int) -> None:
    fn, lib = _fn(source, name, len(tensors))
    stream = torch.cuda.current_stream(tensors[0].device).cuda_stream
    err = fn(*(t.data_ptr() for t in tensors), n, stream)
    if err:
        raise RuntimeError(f"{name}: {lib.cuda_error_string(err).decode()}")


def _check(*ts: torch.Tensor) -> None:
    for t in ts:
        if not (t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()):
            raise ValueError("expected contiguous float32 CUDA tensors")


def vector_add(a: torch.Tensor, b: torch.Tensor, variant: str = "float4") -> torch.Tensor:
    if variant not in VECTOR_ADD_VARIANTS:
        raise ValueError(f"variant must be one of {VECTOR_ADD_VARIANTS}")
    _check(a, b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    c = torch.empty_like(a)
    _launch("vector_add/vector_add.cu", f"vector_add_{variant}", a, b, c, n=a.numel())
    return c


def reduce(x: torch.Tensor, op: str = "sum", variant: str = "warp_shuffle") -> torch.Tensor:
    """Full reduction of `x` to a 0-d tensor."""
    if op not in REDUCE_OPS or variant not in REDUCE_VARIANTS:
        raise ValueError(f"op in {REDUCE_OPS}, variant in {REDUCE_VARIANTS}")
    _check(x)
    out = torch.empty((), device=x.device, dtype=torch.float32)
    _launch("reductions/reductions.cu", f"{op}_{variant}", x, out, n=x.numel())
    return out
