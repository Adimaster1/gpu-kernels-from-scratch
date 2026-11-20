"""Benchmark harness: timing, correctness, and results tables.

Library use:
    from benchmarks.bench import bench, check, write_table

CLI (from the repo root):
    python -m benchmarks.bench                     # every registered kernel
    python -m benchmarks.bench sum max             # by group or kernel name
    python -m benchmarks.bench --list
    python -m benchmarks.bench sum --n 67108864 --out results/sum.md
    python -m benchmarks.bench --format csv --out results/all.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

if __package__ in (None, ""):  # allow `python benchmarks/bench.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------- timing


_l2_flush_buf: torch.Tensor | None = None


def _flush_l2() -> None:
    """Overwrite L2 so a small input isn't served from cache between iterations."""
    global _l2_flush_buf
    if _l2_flush_buf is None:
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        _l2_flush_buf = torch.empty(max(2 * l2, 1 << 22), dtype=torch.uint8, device="cuda")
    _l2_flush_buf.zero_()


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    mean_ms: float
    std_ms: float
    iters: int


def bench_stats(fn: Callable[[], object], warmup: int = 10, iters: int = 100,
                flush_l2: bool = True) -> Timing:
    """Time `fn` on the GPU with CUDA events, one event pair per iteration."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        if flush_l2:
            _flush_l2()  # outside the timed region
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return Timing(
        median_ms=statistics.median(times),
        min_ms=min(times),
        mean_ms=statistics.fmean(times),
        std_ms=statistics.pstdev(times),
        iters=iters,
    )


def bench(fn: Callable[[], object], warmup: int = 10, iters: int = 100,
          flush_l2: bool = True) -> float:
    """Median runtime of `fn` in milliseconds."""
    return bench_stats(fn, warmup, iters, flush_l2).median_ms


# ---------------------------------------------------------------- correctness


@dataclass
class CheckResult:
    ok: bool
    max_abs_err: float

    def __bool__(self) -> bool:
        return self.ok


def check(out: torch.Tensor, ref: torch.Tensor, atol: float = 1e-5,
          rtol: float = 1e-5) -> CheckResult:
    """Compare in float64 with torch.allclose; also report the max abs error."""
    if out.shape != ref.shape:
        return CheckResult(False, float("inf"))
    o, r = out.detach().double(), ref.detach().double()
    ok = torch.allclose(o, r, atol=atol, rtol=rtol)
    err = (o - r).abs().max().item() if o.numel() else 0.0
    return CheckResult(bool(ok), err)


# ----------------------------------------------------------------- hardware


def peak_bandwidth_gbs(device: int = 0) -> float:
    """Theoretical DRAM bandwidth: 2 (DDR) x memory clock x bus width."""
    p = torch.cuda.get_device_properties(device)
    return 2 * p.memory_clock_rate * 1e3 * (p.memory_bus_width / 8) / 1e9


# ---------------------------------------------------------------- reporting

COLUMNS = ["kernel", "variant", "n", "time_ms", "GB/s", "% peak BW", "GFLOP/s",
           "correct", "max_abs_err"]


FLOAT_FORMATS = {"time_ms": ".4f", "GB/s": ".1f", "% peak BW": ".1f", "GFLOP/s": ".2f",
                 "max_abs_err": ".2e"}


def _fmt(v: object, col: str = "") -> str:
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return format(v, FLOAT_FORMATS.get(col, ".4g"))
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def format_table(rows: Sequence[dict], fmt: str = "md", columns: Sequence[str] = COLUMNS) -> str:
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
    if fmt != "md":
        raise ValueError("fmt must be 'md' or 'csv'")
    cells = [[_fmt(r.get(c, ""), c) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
              for i, c in enumerate(columns)]
    numeric = [i > 1 for i in range(len(columns))]
    line = lambda vals: "| " + " | ".join(
        v.rjust(w) if num else v.ljust(w) for v, w, num in zip(vals, widths, numeric)) + " |"
    sep = "|" + "|".join(("-" * (w + 1) + ":") if num else ("-" * (w + 2))
                         for w, num in zip(widths, numeric)) + "|"
    return "\n".join([line(list(columns)), sep, *map(line, cells)]) + "\n"


def write_table(rows: Sequence[dict], path: str | Path | None = None, fmt: str | None = None,
                header: str = "") -> str:
    """Render rows as markdown or CSV; write to `path` if given (format from suffix)."""
    if fmt is None:
        fmt = "csv" if path is not None and str(path).endswith(".csv") else "md"
    text = format_table(rows, fmt)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((header + text) if fmt == "md" else text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------- runner


def run_spec(spec, n: int, warmup: int = 10, iters: int = 100, peak_gbs: float | None = None,
             seed: int = 0) -> dict:
    """Check and time one registered kernel; return a results row."""
    torch.manual_seed(seed)
    inputs = spec.make_inputs(n)
    out = spec.fn(*inputs)
    ref = spec.ref(*inputs)
    atol, rtol = spec.tolerance(*inputs)
    result = check(out, ref, atol=atol, rtol=rtol)

    t = bench_stats(lambda: spec.fn(*inputs), warmup=warmup, iters=iters)
    secs = t.median_ms * 1e-3
    gbs = spec.bytes(n) / secs / 1e9
    peak = peak_gbs or peak_bandwidth_gbs()
    return {
        "kernel": spec.group, "variant": spec.variant, "n": n,
        "time_ms": t.median_ms, "GB/s": gbs, "% peak BW": 100 * gbs / peak,
        "GFLOP/s": spec.flops(n) / secs / 1e9,
        "correct": result.ok, "max_abs_err": result.max_abs_err,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from benchmarks.registry import REGISTRY, select

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="*", help="kernel groups (e.g. sum) or full names (sum/warp_shuffle)")
    ap.add_argument("--n", type=lambda s: int(eval(s, {}, {})), default=1 << 24,
                    help="number of elements (expressions like 2**24 allowed); default 2**24")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--peak-gbs", type=float, default=None, help="override theoretical peak BW")
    ap.add_argument("--format", choices=["md", "csv"], default=None)
    ap.add_argument("--out", type=Path, default=None, help="write table to this file (.md or .csv)")
    ap.add_argument("--list", action="store_true", help="list registered kernels and exit")
    args = ap.parse_args(argv)

    if args.list:
        for s in REGISTRY:
            print(s.name)
        return 0
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    specs = select(args.names)
    peak = args.peak_gbs or peak_bandwidth_gbs()
    dev = torch.cuda.get_device_properties(0)
    header = (f"GPU: {dev.name} | {dev.multi_processor_count} SMs | "
              f"theoretical peak DRAM BW {peak:.1f} GB/s | torch {torch.__version__} "
              f"(CUDA {torch.version.cuda}) | median of {args.iters} iters, L2 flushed\n\n")
    print(header, end="", file=sys.stderr)

    rows = []
    for spec in specs:
        row = run_spec(spec, args.n, args.warmup, args.iters, peak)
        rows.append(row)
        print(f"  {spec.name:<28} {row['time_ms']:9.4f} ms  {row['GB/s']:8.1f} GB/s  "
              f"{row['% peak BW']:5.1f}%  {'ok' if row['correct'] else 'WRONG'}", file=sys.stderr)

    fmt = args.format or ("csv" if args.out and args.out.suffix == ".csv" else "md")
    print(write_table(rows, args.out, fmt, header=header), end="")
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)
    return 0 if all(r["correct"] for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
