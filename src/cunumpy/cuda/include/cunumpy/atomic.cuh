// cunumpy/atomic.cuh: atomic accumulation helpers for CUDA kernels.
//
// Many threads adding into a few cells (particle-to-grid accumulation) must
// use atomics or lose updates. These helpers wrap atomicAdd for double and
// float, and index C-contiguous 2D/3D arrays passed as bare pointers plus
// their trailing extents.
//
// atomicAdd(double*, double) is a hardware instruction from compute
// capability 6.0 (sm_60) on; for older devices a compare-and-swap loop is
// used, which is correct but slow.
//
// The header directory is added to every CudaKernel's NVRTC options, so:
//     #include <cunumpy/atomic.cuh>

#ifndef CUNUMPY_ATOMIC_CUH
#define CUNUMPY_ATOMIC_CUH

// Add v to *p atomically; returns the old value of *p.
__device__ __forceinline__ double cunumpy_atomic_add(double* p, double v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
    unsigned long long* address = reinterpret_cast<unsigned long long*>(p);
    unsigned long long old = *address;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed,
                        __double_as_longlong(v + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    return atomicAdd(p, v);
#endif
}

__device__ __forceinline__ float cunumpy_atomic_add(float* p, float v)
{
    return atomicAdd(p, v);
}

// data[i, j] += v for a C-contiguous array of shape (n0, n1).
__device__ __forceinline__ double cunumpy_atomic_add_2d(
    double* data, long long n1, long long i, long long j, double v)
{
    return cunumpy_atomic_add(data + i * n1 + j, v);
}

__device__ __forceinline__ float cunumpy_atomic_add_2d(
    float* data, long long n1, long long i, long long j, float v)
{
    return cunumpy_atomic_add(data + i * n1 + j, v);
}

// data[i, j, k] += v for a C-contiguous array of shape (n0, n1, n2).
__device__ __forceinline__ double cunumpy_atomic_add_3d(
    double* data, long long n1, long long n2,
    long long i, long long j, long long k, double v)
{
    return cunumpy_atomic_add(data + (i * n1 + j) * n2 + k, v);
}

__device__ __forceinline__ float cunumpy_atomic_add_3d(
    float* data, long long n1, long long n2,
    long long i, long long j, long long k, float v)
{
    return cunumpy_atomic_add(data + (i * n1 + j) * n2 + k, v);
}

#endif  // CUNUMPY_ATOMIC_CUH
