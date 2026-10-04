// cunumpy/atomic.cuh: atomic accumulation helpers for CUDA kernels.
//
// Many threads adding into a few cells (particle-to-grid accumulation) must
// use atomics or lose updates. These helpers wrap atomicAdd for double, float,
// signed/unsigned 32-bit and 64-bit integers, and index C-contiguous 2D/3D arrays
// passed as bare pointers plus
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

__device__ __forceinline__ int cunumpy_atomic_add(int* p, int v)
{
    return atomicAdd(p, v);
}

__device__ __forceinline__ unsigned int cunumpy_atomic_add(unsigned int* p, unsigned int v)
{
    return atomicAdd(p, v);
}

__device__ __forceinline__ unsigned long long cunumpy_atomic_add(
    unsigned long long* p, unsigned long long v)
{
    return atomicAdd(p, v);
}

// Signed 64-bit addition through CUDA's unsigned 64-bit atomic. Arithmetic
// follows the unsigned instruction's modulo-2^64 behavior.
__device__ __forceinline__ long long cunumpy_atomic_add(long long* p, long long v)
{
    return (long long)atomicAdd(reinterpret_cast<unsigned long long*>(p),
                               (unsigned long long)v);
}

// Typed indexed overloads also serve integer counters and occupancy arrays.
template <class T>
__device__ __forceinline__ T cunumpy_atomic_add_2d(
    T* data, long long n1, long long i, long long j, T v)
{
    return cunumpy_atomic_add(data + i * n1 + j, v);
}

template <class T>
__device__ __forceinline__ T cunumpy_atomic_add_3d(
    T* data, long long n1, long long n2,
    long long i, long long j, long long k, T v)
{
    return cunumpy_atomic_add(data + (i * n1 + j) * n2 + k, v);
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
