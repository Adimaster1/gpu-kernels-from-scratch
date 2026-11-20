// Sum and max reductions over a float32 vector, from naive to fast.
//
//   v0 naive_atomic    every thread atomically updates one global value
//   v1 smem_tree       one element per thread, shared-memory tree per block,
//                      one atomic per block
//   v2 smem_tree_gs    grid-stride loop first (each thread sums many elements
//                      in a register), then the same shared-memory tree
//   v3 warp_shuffle    grid-stride + float4 loads, warp-level __shfl_down_sync
//                      reduction, only 32 floats of shared memory per block
//
// Every version is templated on an Op (SumOp / MaxOp), so sum and max share
// the exact same code. A reduction is memory-bound (4 bytes per element), so
// the target is a high fraction of peak DRAM bandwidth.
#include "../common.cuh"
#include <math_constants.h>

constexpr int BLOCK = 256;
constexpr unsigned FULL_MASK = 0xffffffffu;

// Float atomic max built from integer atomics. For non-negative floats the
// IEEE bit pattern orders like a signed int; for negative floats it orders in
// reverse like an unsigned int. (NaN is not handled.)
__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    if (v >= 0.0f)
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
    else
        atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(v));
}

struct SumOp {
    __device__ __forceinline__ static float identity() { return 0.0f; }
    __device__ __forceinline__ static float combine(float a, float b) { return a + b; }
    __device__ __forceinline__ static void atomic(float* p, float v) { atomicAdd(p, v); }
};

struct MaxOp {
    __device__ __forceinline__ static float identity() { return -CUDART_INF_F; }
    __device__ __forceinline__ static float combine(float a, float b) { return fmaxf(a, b); }
    __device__ __forceinline__ static void atomic(float* p, float v) { atomic_max_float(p, v); }
};

template <class Op>
__global__ void init_kernel(float* out) { *out = Op::identity(); }

// ---------------------------------------------------------------- v0 -------
// One global atomic per element. All n atomics hit the same address, so they
// serialize in L2: this is limited by atomic throughput, not bandwidth.
template <class Op>
__global__ void reduce_naive_atomic(const float* __restrict__ x, float* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) Op::atomic(out, x[i]);
}

// ---------------------------------------------------------------- v1 -------
// Classic shared-memory tree with sequential addressing (no bank conflicts,
// no divergent `tid % (2*s)`). log2(BLOCK) steps, each with a __syncthreads.
// Weaknesses: half the threads idle after the first step, one element per
// thread means a huge grid and one atomic per 256 elements.
template <class Op>
__global__ void reduce_smem_tree(const float* __restrict__ x, float* out, long long n) {
    __shared__ float s[BLOCK];
    unsigned t = threadIdx.x;
    long long i = blockIdx.x * (long long)BLOCK + t;
    s[t] = (i < n) ? x[i] : Op::identity();
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] = Op::combine(s[t], s[t + stride]);
        __syncthreads();
    }
    if (t == 0) Op::atomic(out, s[0]);
}

// ---------------------------------------------------------------- v2 -------
// Same tree, but each thread first accumulates many elements in a register
// with a grid-stride loop. The grid is just big enough to fill the GPU, so
// the tree + atomic overhead is paid ~100 times instead of ~65k times.
template <class Op>
__global__ void reduce_smem_tree_gs(const float* __restrict__ x, float* out, long long n) {
    __shared__ float s[BLOCK];
    unsigned t = threadIdx.x;
    long long stride = (long long)gridDim.x * BLOCK;
    float acc = Op::identity();
    for (long long i = blockIdx.x * (long long)BLOCK + t; i < n; i += stride)
        acc = Op::combine(acc, x[i]);
    s[t] = acc;
    __syncthreads();
    for (int st = BLOCK / 2; st > 0; st >>= 1) {
        if (t < st) s[t] = Op::combine(s[t], s[t + st]);
        __syncthreads();
    }
    if (t == 0) Op::atomic(out, s[0]);
}

// ---------------------------------------------------------------- v3 -------
// Threads in a warp exchange registers directly with __shfl_down_sync, so the
// in-warp part of the tree needs no shared memory and no __syncthreads.
template <class Op>
__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = Op::combine(v, __shfl_down_sync(FULL_MASK, v, offset));
    return v;
}

template <class Op>
__global__ void reduce_warp_shuffle(const float* __restrict__ x, float* out,
                                    long long n, long long n4) {
    long long tid = blockIdx.x * (long long)BLOCK + threadIdx.x;
    long long stride = (long long)gridDim.x * BLOCK;

    // 1) Grid-stride accumulate with 16-byte loads.
    float acc = Op::identity();
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (long long i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        acc = Op::combine(acc, Op::combine(Op::combine(v.x, v.y), Op::combine(v.z, v.w)));
    }
    for (long long i = n4 * 4 + tid; i < n; i += stride) acc = Op::combine(acc, x[i]);

    // 2) Reduce within each warp (32 -> 1) using register shuffles.
    acc = warp_reduce<Op>(acc);

    // 3) One value per warp goes to shared memory; warp 0 reduces those.
    __shared__ float warp_vals[BLOCK / 32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_vals[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < BLOCK / 32) ? warp_vals[lane] : Op::identity();
        acc = warp_reduce<Op>(acc);
        if (lane == 0) Op::atomic(out, acc);
    }
}

// ------------------------------------------------------------ launchers ----
template <class Kernel>
static int grid_for(Kernel kernel, long long work) {
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, BLOCK, 0);
    return max(1, min(ceil_div(work, BLOCK), num_sms() * blocks_per_sm));
}

template <class Op>
static int launch_naive_atomic(const float* x, float* out, long long n, cudaStream_t s) {
    init_kernel<Op><<<1, 1, 0, s>>>(out);
    if (n > 0) reduce_naive_atomic<Op><<<ceil_div(n, BLOCK), BLOCK, 0, s>>>(x, out, n);
    return cudaGetLastError();
}

template <class Op>
static int launch_smem_tree(const float* x, float* out, long long n, cudaStream_t s) {
    init_kernel<Op><<<1, 1, 0, s>>>(out);
    if (n > 0) reduce_smem_tree<Op><<<ceil_div(n, BLOCK), BLOCK, 0, s>>>(x, out, n);
    return cudaGetLastError();
}

template <class Op>
static int launch_smem_tree_gs(const float* x, float* out, long long n, cudaStream_t s) {
    init_kernel<Op><<<1, 1, 0, s>>>(out);
    if (n > 0) {
        int grid = grid_for(reduce_smem_tree_gs<Op>, n);
        reduce_smem_tree_gs<Op><<<grid, BLOCK, 0, s>>>(x, out, n);
    }
    return cudaGetLastError();
}

template <class Op>
static int launch_warp_shuffle(const float* x, float* out, long long n, cudaStream_t s) {
    init_kernel<Op><<<1, 1, 0, s>>>(out);
    if (n > 0) {
        long long n4 = is_aligned16(x) ? n / 4 : 0;
        int grid = grid_for(reduce_warp_shuffle<Op>, n4 > 0 ? n4 : n);
        reduce_warp_shuffle<Op><<<grid, BLOCK, 0, s>>>(x, out, n, n4);
    }
    return cudaGetLastError();
}

#define DEFINE_LAUNCHERS(variant)                                                        \
    EXPORT int sum_##variant(const float* x, float* out, long long n, cudaStream_t s) { \
        return launch_##variant<SumOp>(x, out, n, s);                                    \
    }                                                                                    \
    EXPORT int max_##variant(const float* x, float* out, long long n, cudaStream_t s) { \
        return launch_##variant<MaxOp>(x, out, n, s);                                    \
    }

DEFINE_LAUNCHERS(naive_atomic)
DEFINE_LAUNCHERS(smem_tree)
DEFINE_LAUNCHERS(smem_tree_gs)
DEFINE_LAUNCHERS(warp_shuffle)
