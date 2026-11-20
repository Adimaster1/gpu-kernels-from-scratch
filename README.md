
# GPU Kernels from Scratch

Implementing and optimizing GPU kernels using CUDA C++ and Triton.

The goal of this project is to understand GPU architecture,
parallel programming, memory hierarchy, kernel optimization,
and performance profiling by implementing fundamental
deep learning operations from scratch.

## Implementations

### 1. Matrix Multiplication
- [ ] Naive CUDA
- [ ] Tiled CUDA
- [ ] Shared-memory optimization
- [ ] Coalesced memory access
- [ ] Tensor Core implementation
- [ ] Triton implementation

### 2. Fundamental Kernels
- [x] Vector add (naive, float4 grid-stride): 94.5% of peak BW
- [ ] Softmax
- [ ] LayerNorm
- [x] Sum reduction (naive atomic → smem tree → grid-stride → warp shuffle): 95.5% of peak BW
- [x] Max reduction (same progression): 95.7% of peak BW

See [docs/phase1-reductions.md](docs/phase1-reductions.md) for the analysis.

### 3. Attention
- [ ] Basic attention
- [ ] Tiled attention
- [ ] Online softmax
- [ ] FlashAttention-style fused attention

## Running

```bash
python -m pytest                                    # correctness for every kernel
python -m benchmarks.bench --list                   # registered kernels
python -m benchmarks.bench                          # benchmark everything (n = 2**24)
python -m benchmarks.bench sum max --n 2**26        # select groups / sizes
python -m benchmarks.bench --out results/phase1.md  # markdown table (.csv -> CSV)
```

CUDA kernels in `kernels/cuda/` are compiled to a DLL by nvcc on first use
(cached in `kernels/cuda/build/`; MSVC is located automatically via vcvars) and
called from Python with ctypes. To add a kernel, add an `EXPORT` launcher in a
`.cu` file, a wrapper in `kernels/cuda/ops.py`, and a `KernelSpec` in
`benchmarks/registry.py`. The benchmark CLI and the tests then pick it up.

The harness (`benchmarks/bench.py`) times with CUDA events (warmup, L2 flush
between iterations, median), checks against a float64 reference with
`torch.allclose`, and reports time, GB/s, % of theoretical peak bandwidth,
GFLOP/s, and correctness.

## Performance Analysis

Every implementation includes:
- Correctness tests
- Benchmark comparisons
- Nsight profiling
- Performance analysis
- Optimization notes

## Technology Stack

- CUDA C++
- Python
- PyTorch
- Triton
- NVIDIA Nsight Compute
- NVIDIA Nsight Systems

## Hardware

GPU: RTX 3050 Laptop 6GB  
CUDA Version: CUDA 13.2  
Driver Version: driver 610.91

## Repository Structure

- `kernels/` - CUDA and Triton implementations
- `benchmarks/` - Benchmarking scripts
- `tests/` - Correctness tests
- `results/` - Performance results
- `docs/` - GPU architecture and optimization notes
- `notebooks/` - Performance visualization

## Project Philosophy

Start with a simple implementation, understand its bottlenecks,
optimize one thing at a time, and measure every improvement.

All performance results are measured on the hardware
documented above.
