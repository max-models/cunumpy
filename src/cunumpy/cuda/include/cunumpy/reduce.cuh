// cunumpy/reduce.cuh: warp- and block-level reductions for CUDA kernels.
//
// Diagnostics computed inside a kernel (kinetic energy, momentum, total
// charge, a maximum velocity for the CFL condition) and accumulation that
// combines values in a block before writing to global memory all need the
// same reduction: shuffles within each warp, then shared memory across the
// warps of a block. These helpers implement it once:
//
//     #include <cunumpy/reduce.cuh>
//
//     extern "C" __global__ void kinetic_energy(const double* v, long long n,
//                                               double mass, double* energy) {
//         long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
//         double e = i < n ? 0.5 * mass * v[i] * v[i] : 0.0;  // no early return
//         cunumpy_block_sum_to(energy, e);  // one atomic add per block
//     }
//
// Rules:
// * Every thread of the block must call a block function, and all 32 lanes of
//   the warp a warp function: do not return early. Threads without a value
//   pass the identity (0 for a sum, the largest value for a minimum, ...).
// * The number of threads per block must be a multiple of 32 (CudaKernel's
//   default block size is 128).
// * The result is returned to every thread (warp functions: every lane).
// * Supported types are those of the shuffle intrinsics: int, unsigned,
//   long long, unsigned long long, float, double.
//
// The warp size is 32, as on all NVIDIA GPUs.
//
// The header directory is added to every CudaKernel's NVRTC options.

#ifndef CUNUMPY_REDUCE_CUH
#define CUNUMPY_REDUCE_CUH

#include "cunumpy/atomic.cuh"

#define CUNUMPY_WARP_SIZE 32
#define CUNUMPY_FULL_WARP_MASK 0xffffffffu

// The binary operations; fill() gives the value of the lanes of the last
// step that have no partial result of their own.
struct cunumpy_sum_op {
    template <class T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
    template <class T>
    __device__ __forceinline__ static T fill(const T*) { return T(0); }
};

struct cunumpy_min_op {
    template <class T>
    __device__ __forceinline__ T operator()(T a, T b) const { return b < a ? b : a; }
    template <class T>
    __device__ __forceinline__ static T fill(const T* partial) { return partial[0]; }
};

struct cunumpy_max_op {
    template <class T>
    __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? b : a; }
    template <class T>
    __device__ __forceinline__ static T fill(const T* partial) { return partial[0]; }
};

// Linear index of the thread in its block, and the number of threads per block
// (for 1D to 3D blocks).
__device__ __forceinline__ int cunumpy_block_thread()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ __forceinline__ int cunumpy_block_threads()
{
    return blockDim.x * blockDim.y * blockDim.z;
}

// Reduce v over the 32 lanes of the warp; every lane gets the result.
template <class T, class Op>
__device__ __forceinline__ T cunumpy_warp_reduce(T v, Op op)
{
    for (int mask = CUNUMPY_WARP_SIZE / 2; mask > 0; mask /= 2) {
        v = op(v, __shfl_xor_sync(CUNUMPY_FULL_WARP_MASK, v, mask));
    }
    return v;
}

template <class T>
__device__ __forceinline__ T cunumpy_warp_sum(T v) { return cunumpy_warp_reduce(v, cunumpy_sum_op()); }

template <class T>
__device__ __forceinline__ T cunumpy_warp_min(T v) { return cunumpy_warp_reduce(v, cunumpy_min_op()); }

template <class T>
__device__ __forceinline__ T cunumpy_warp_max(T v) { return cunumpy_warp_reduce(v, cunumpy_max_op()); }

// Reduce v over all threads of the block; every thread gets the result.
// Uses 32 values of static shared memory per type and synchronizes the block
// (__syncthreads) three times; it may be called several times in a kernel.
template <class T, class Op>
__device__ T cunumpy_block_reduce(T v, Op op)
{
    __shared__ T partial[CUNUMPY_WARP_SIZE];
    const int thread = cunumpy_block_thread();
    const int lane = thread % CUNUMPY_WARP_SIZE;
    const int warp = thread / CUNUMPY_WARP_SIZE;
    const int n_warps = (cunumpy_block_threads() + CUNUMPY_WARP_SIZE - 1) / CUNUMPY_WARP_SIZE;

    v = cunumpy_warp_reduce(v, op);
    __syncthreads();  // a previous call may still be reading partial
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < n_warps ? partial[lane] : Op::fill(partial);
        v = cunumpy_warp_reduce(v, op);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

template <class T>
__device__ T cunumpy_block_sum(T v) { return cunumpy_block_reduce(v, cunumpy_sum_op()); }

template <class T>
__device__ T cunumpy_block_min(T v) { return cunumpy_block_reduce(v, cunumpy_min_op()); }

template <class T>
__device__ T cunumpy_block_max(T v) { return cunumpy_block_reduce(v, cunumpy_max_op()); }

// *out += sum of v over the block, with one atomic add per block (thread 0).
// For a sum over the whole grid, zero *out before the launch.
__device__ __forceinline__ void cunumpy_block_sum_to(double* out, double v)
{
    const double total = cunumpy_block_sum(v);
    if (cunumpy_block_thread() == 0) cunumpy_atomic_add(out, total);
}

__device__ __forceinline__ void cunumpy_block_sum_to(float* out, float v)
{
    const float total = cunumpy_block_sum(v);
    if (cunumpy_block_thread() == 0) cunumpy_atomic_add(out, total);
}

__device__ __forceinline__ void cunumpy_block_sum_to(int* out, int v)
{
    const int total = cunumpy_block_sum(v);
    if (cunumpy_block_thread() == 0) atomicAdd(out, total);
}

__device__ __forceinline__ void cunumpy_block_sum_to(unsigned long long* out, unsigned long long v)
{
    const unsigned long long total = cunumpy_block_sum(v);
    if (cunumpy_block_thread() == 0) atomicAdd(out, total);
}

#endif  // CUNUMPY_REDUCE_CUH
