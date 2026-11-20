// Shared helpers for kernels that are compiled into a DLL/.so and called from
// Python via ctypes (see kernels/cuda/_build.py and kernels/cuda/ops.py).
//
// Convention: every exported launcher returns an int cudaError_t (0 = success)
// and takes the caller's cudaStream_t, so it runs on PyTorch's current stream.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Each DLL exports this so Python can turn an error code into a message.
EXPORT const char* cuda_error_string(int err) {
    return cudaGetErrorString(static_cast<cudaError_t>(err));
}

static inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

static inline int ceil_div(long long a, long long b) {
    return static_cast<int>((a + b - 1) / b);
}

// Number of SMs on the current device (cached after the first call).
static inline int num_sms() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}
