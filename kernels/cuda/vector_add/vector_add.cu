// Vector add: C = A + B (float32).
//
// Memory-bound: 2 loads + 1 store per element = 12 bytes, 1 FLOP.
// The roofline ceiling is therefore peak DRAM bandwidth.
#include "../common.cuh"

constexpr int BLOCK = 256;

// v0: one thread per element. Already coalesced, so this is close to optimal;
// the main costs are index math and one 4-byte transaction per thread.
__global__ void vector_add_naive_kernel(const float* __restrict__ a,  // __global__ declares it has GPU kernel
                                        const float* __restrict__ b,
                                        float* __restrict__ c, long long n) {
                                            //__restrict__ is compiler optimization trick to avoid overlap memory regions
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// v1: grid-stride loop with 16-byte (float4) vector loads/stores. -> purpose : threads are no longer restricted to single element
// Fewer instructions per byte moved, and a fixed-size grid sized to the GPU.
__global__ void vector_add_float4_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c, long long n,
                                         long long n4) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;// identical to v0
    long long stride = (long long)gridDim.x * blockDim.x; // smaller grid instead of giving enough blocks

    //2nd Optimization
    // float4 is cuda vector type containing 4 floats [x,y,z,w]
    const float4* a4 = reinterpret_cast<const float4*>(a); // pointer doesn't change but grouping change
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (long long i = tid; i < n4; i += stride) {
        float4 x = a4[i], y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    // Scalar tail (n not a multiple of 4, or unaligned pointers -> n4 == 0). -> V0 method for remaining threads
    for (long long i = n4 * 4 + tid; i < n; i += stride) c[i] = a[i] + b[i];
}

EXPORT int vector_add_naive(const float* a, const float* b, float* c, //CPU wrapper function for v0
                            long long n, cudaStream_t stream) {
    if (n <= 0) return 0;
    vector_add_naive_kernel<<<ceil_div(n, BLOCK), BLOCK, 0, stream>>>(a, b, c, n); //function call the gpu kernel
    return cudaGetLastError();
}

EXPORT int vector_add_float4(const float* a, const float* b, float* c, //CPU wrapper function for v1
                             long long n, cudaStream_t stream) {
    if (n <= 0) return 0;
    bool aligned = is_aligned16(a) && is_aligned16(b) && is_aligned16(c);
    long long n4 = aligned ? n / 4 : 0;
    long long work = aligned ? n4 + (n - n4 * 4) : n;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                  vector_add_float4_kernel, BLOCK, 0);
    int grid = min(ceil_div(work, BLOCK), num_sms() * blocks_per_sm);
    vector_add_float4_kernel<<<grid, BLOCK, 0, stream>>>(a, b, c, n, n4); //function call the gpu kernel
    return cudaGetLastError();
}
