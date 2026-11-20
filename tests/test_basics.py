"""Correctness tests for every registered kernel (Phase 0/1)."""
import pytest
import torch

from benchmarks.bench import check, format_table, write_table
from benchmarks.registry import REGISTRY

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Odd sizes exercise bounds checks and the scalar tail after float4 loads.
SIZES = [1, 3, 255, 256, 257, 4099, 1 << 20, (1 << 22) + 7]
CUSTOM = [s for s in REGISTRY if s.variant != "torch"]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("spec", CUSTOM, ids=lambda s: s.name)
def test_matches_reference(spec, n):
    torch.manual_seed(n)
    inputs = spec.make_inputs(n)
    atol, rtol = spec.tolerance(*inputs)
    res = check(spec.fn(*inputs), spec.ref(*inputs), atol=atol, rtol=rtol)
    assert res, f"{spec.name} n={n}: max abs err {res.max_abs_err}"


@pytest.mark.parametrize("spec", CUSTOM, ids=lambda s: s.name)
def test_unaligned_input(spec):
    """Slicing off one element breaks 16-byte alignment; float4 paths must fall back."""
    torch.manual_seed(0)
    inputs = tuple(t[1:] for t in spec.make_inputs(10_001))
    atol, rtol = spec.tolerance(*inputs)
    assert check(spec.fn(*inputs), spec.ref(*inputs), atol=atol, rtol=rtol)


@pytest.mark.parametrize("op", ["max"])
def test_max_all_negative(op):
    from kernels.cuda import ops
    x = -torch.rand(100_003, device="cuda") - 1.0
    for v in ops.REDUCE_VARIANTS:
        assert ops.reduce(x, op, v).item() == x.max().item(), v


def test_check_detects_mismatch():
    a = torch.zeros(4)
    assert check(a, a)
    assert not check(a + 1, a)
    assert not check(a, torch.zeros(5))


def test_table_writers(tmp_path):
    rows = [{"kernel": "k", "variant": "v", "n": 10, "time_ms": 0.5, "GB/s": 1.0,
             "% peak BW": 50.0, "GFLOP/s": 2.0, "correct": True, "max_abs_err": 0.0}]
    md = format_table(rows, "md")
    assert md.splitlines()[0].startswith("| kernel")
    write_table(rows, tmp_path / "r.csv")
    assert (tmp_path / "r.csv").read_text().splitlines()[1].startswith("k,v,10,")
