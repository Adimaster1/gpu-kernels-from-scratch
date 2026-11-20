
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
- [ ] Softmax
- [ ] LayerNorm
- [ ] Sum reduction
- [ ] Max reduction

### 3. Attention
- [ ] Basic attention
- [ ] Tiled attention
- [ ] Online softmax
- [ ] FlashAttention-style fused attention

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

GPU: To be documented  
CUDA Version: To be documented  
Driver Version: To be documented

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
