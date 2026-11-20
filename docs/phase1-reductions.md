# Phase 1: vector add and reductions

All numbers: RTX 3050 6GB Laptop (GA107, 20 SMs, 96-bit GDDR6), theoretical
peak DRAM bandwidth **131.7 GB/s** (2 x 5486 MHz x 12 B). Float32, n = 2^24,
median of 100 iterations with L2 flushed between iterations. Full tables:
[`results/phase1.md`](../results/phase1.md), [`results/phase1_reductions_64M.md`](../results/phase1_reductions_64M.md).

Reproduce: `python -m benchmarks.bench --out results/phase1.md`

## Roofline

Both kernels do ~1 FLOP per element while moving 12 B (vector add) or 4 B
(reduction). Arithmetic intensity is 0.08 to 0.25 FLOP/B, far below the ridge
point, so the only ceiling that matters is DRAM bandwidth. Success = % of peak BW.

## Vector add

| variant | GB/s | % peak |
|---|---:|---:|
| naive (1 thread / element) | 124.4 | 94.5 |
| float4 grid-stride | 121.3 | 92.1 |
| `torch.add` | 123.6 | 93.9 |

Plain one-thread-per-element is already coalesced and saturates DRAM, and
`float4` doesn't help here. Vectorizing reduces instruction count, which only
matters when the kernel is issue-bound rather than memory-bound.

## Sum / max reduction progression

| variant | sum GB/s | sum % peak | max GB/s | max % peak |
|---|---:|---:|---:|---:|
| v0 naive_atomic | 2.0 | 1.5 | 91.8 | 69.7 |
| v1 smem_tree | 72.5 | 55.1 | 72.2 | 54.8 |
| v2 smem_tree_gs (grid-stride + tree) | 126.0 | 95.7 | 125.8 | 95.5 |
| v3 warp_shuffle (grid-stride + float4 + shfl) | 125.8 | 95.5 | 126.0 | 95.7 |
| `torch.sum` / `torch.amax` | 125.3 | 95.2 | 125.3 | 95.2 |

**Target (70%+ of peak) met** by v2 and v3. Both match PyTorch.

What each step teaches:

- **v0 naive_atomic**: n atomics on a single address serialize in L2 (~0.5
  G atomics/s). Float `atomicAdd` can't be combined, so sum is ~65x slower than
  memory speed. Max is surprisingly fast (70%). It is built from *integer*
  `atomicMax`/`atomicMin`, and the compiler can aggregate same-address integer
  atomics within a warp (one L2 op per warp instead of 32). Check the SASS with
  `cuobjdump -sass` to confirm.
- **v1 smem_tree**: sequential-addressing tree in shared memory, one atomic per
  256 elements. It stalls at ~55%. Each thread issues a single 4-byte load and
  then waits at `__syncthreads`, so there are too few bytes in flight to cover
  DRAM latency, and 65k short-lived blocks pay launch and tree overhead each.
- **v2 smem_tree_gs**: same tree, but a grid sized to occupancy (~120 blocks)
  and each thread first accumulates ~550 elements in a register. Many loads stay
  in flight, and the tree and atomic run 120 times instead of 65k. This is the
  step that reaches bandwidth.
- **v3 warp_shuffle**: `__shfl_down_sync` does the in-warp tree in registers
  (no shared memory and no barriers for 5 of 8 levels), and float4 loads cut load
  instructions 4x. At this size it ties v2 because both are DRAM-bound. The gain
  is fewer instructions and syncs, which shows up in Nsight Compute (issue
  slots, barrier stalls) rather than in wall time.

## Numerical note

The sum tolerance is `1e-5 * sum(|x|)`, the standard backward-error scale for
floating-point summation. A tolerance relative to the result would be
meaningless for zero-mean data. v0 accumulates 16M values serially into one
float and has an error of ~0.47. The tree-based versions are ~1e-3, and the
reference is computed in float64.
