"""Every benchmarkable kernel, with its inputs, reference, and cost model.

To add a kernel: append a KernelSpec to REGISTRY. The CLI in bench.py and the
tests in tests/ pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from kernels.cuda import ops

F32 = 4  # bytes


@dataclass(frozen=True)
class KernelSpec:
    group: str                                   # e.g. "sum"
    variant: str                                 # e.g. "warp_shuffle" or "torch"
    make_inputs: Callable[[int], tuple]          # n -> inputs
    fn: Callable[..., torch.Tensor]              # kernel under test
    ref: Callable[..., torch.Tensor]             # trusted reference
    bytes: Callable[[int], int]                  # DRAM bytes moved (ideal)
    flops: Callable[[int], int]
    tolerance: Callable[..., tuple[float, float]] = field(default=lambda *a: (1e-5, 1e-5))

    @property
    def name(self) -> str:
        return f"{self.group}/{self.variant}"


def _vec(n: int) -> tuple:
    return (torch.rand(n, device="cuda") * 2 - 1,)


def _two_vecs(n: int) -> tuple:
    return (torch.rand(n, device="cuda") * 2 - 1, torch.rand(n, device="cuda") * 2 - 1)


def _sum_tol(x: torch.Tensor) -> tuple[float, float]:
    # Floating-point summation error scales with sum(|x|), not with the result
    # (which can be ~0 for zero-mean data). Allow 1e-5 relative to that.
    return 1e-5 * x.abs().sum().item() + 1e-6, 1e-5


REGISTRY: list[KernelSpec] = []

# ------------------------------------------------------------------ vector add
for v in (*ops.VECTOR_ADD_VARIANTS, "torch"):
    REGISTRY.append(KernelSpec(
        group="vector_add", variant=v, make_inputs=_two_vecs,
        fn=torch.add if v == "torch" else (lambda a, b, v=v: ops.vector_add(a, b, v)),
        ref=lambda a, b: a.double() + b.double(),
        bytes=lambda n: 3 * F32 * n, flops=lambda n: n,
        tolerance=lambda a, b: (1e-6, 1e-6),
    ))

# ------------------------------------------------------------------ reductions
_TORCH_REDUCE = {"sum": torch.sum, "max": torch.amax}
for op in ops.REDUCE_OPS:
    for v in (*ops.REDUCE_VARIANTS, "torch"):
        REGISTRY.append(KernelSpec(
            group=op, variant=v, make_inputs=_vec,
            fn=_TORCH_REDUCE[op] if v == "torch" else (lambda x, op=op, v=v: ops.reduce(x, op, v)),
            ref=(lambda x: x.double().sum()) if op == "sum" else (lambda x: x.max()),
            bytes=lambda n: F32 * n, flops=lambda n: n,
            tolerance=_sum_tol if op == "sum" else (lambda x: (0.0, 0.0)),
        ))


def select(names: Sequence[str]) -> list[KernelSpec]:
    """Pick specs by group ("sum") or full name ("sum/warp_shuffle"); empty = all."""
    if not names:
        return list(REGISTRY)
    chosen = [s for s in REGISTRY if s.group in names or s.name in names]
    unknown = set(names) - {s.group for s in REGISTRY} - {s.name for s in REGISTRY}
    if unknown:
        raise SystemExit(f"unknown kernel(s): {', '.join(sorted(unknown))}; see --list")
    return chosen
