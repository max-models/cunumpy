// Prefix sums for compaction and binning. Scan order is increasing linear
// thread index (x fastest), or increasing lane index among a warp mask's bits.
// Every named lane must call a warp function with the same nonzero mask;
// every thread must call a block function, including padding threads (value 0).
// Supports CUDA shuffle arithmetic types, arbitrary masks and partial warps.
// Block scans use 32 shared values per type; calls must be collective.
// Not yet ported to HIP/ROCm, see cunumpy/reduce.cuh.
#ifndef CUNUMPY_SCAN_CUH
#define CUNUMPY_SCAN_CUH

#include "cunumpy/reduce.cuh"

template <class T>
__device__ __forceinline__ T cunumpy_warp_inclusive_sum(
    T v, unsigned mask = CUNUMPY_FULL_WARP_MASK)
{
    const int lane = cunumpy_block_thread() % CUNUMPY_WARP_SIZE;
    if ((mask & (mask + 1u)) == 0u) {
        // Full warp or contiguous prefix of lanes.
        for (int offset = 1; offset < CUNUMPY_WARP_SIZE; offset *= 2) {
            T previous = __shfl_up_sync(mask, v, offset);
            if (lane >= offset) v += previous;
        }
        return v;
    }
    T result = T(0);
    unsigned remaining = mask;
    while (remaining) {
        const int source = __ffs(remaining) - 1;
        T item = __shfl_sync(mask, v, source);
        if (source <= lane) result += item;
        remaining &= remaining - 1;
    }
    return result;
}

template <class T>
__device__ __forceinline__ T cunumpy_warp_exclusive_sum(
    T v, unsigned mask = CUNUMPY_FULL_WARP_MASK)
{
    const int lane = cunumpy_block_thread() % CUNUMPY_WARP_SIZE;
    const T inclusive = cunumpy_warp_inclusive_sum(v, mask);
    const unsigned before = mask & ((1u << lane) - 1u);
    // Shuffle for every participating lane, including the first (whose result
    // is discarded); subtraction would lose precision for large current values.
    const int source = before ? 31 - __clz(before) : lane;
    const T previous = __shfl_sync(mask, inclusive, source);
    return before ? previous : T(0);
}

template <class T, bool Exclusive>
__device__ T cunumpy_block_scan_sum(T v)
{
    __shared__ T partial[CUNUMPY_WARP_SIZE];
    const int thread = cunumpy_block_thread();
    const int lane = thread % CUNUMPY_WARP_SIZE;
    const int warp = thread / CUNUMPY_WARP_SIZE;
    const int n_warps = (cunumpy_block_threads() + CUNUMPY_WARP_SIZE - 1) / CUNUMPY_WARP_SIZE;
    const unsigned mask = cunumpy_block_warp_mask();
    const T inclusive = cunumpy_warp_inclusive_sum(v, mask);
    const unsigned before = mask & ((1u << lane) - 1u);
    const int source = before ? 31 - __clz(before) : lane;
    const T previous = __shfl_sync(mask, inclusive, source);
    const T local = Exclusive ? (before ? previous : T(0)) : inclusive;
    __syncthreads();  // protect readers from a previous scan invocation
    if (lane == 31 - __clz(mask)) partial[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        const T total = cunumpy_warp_inclusive_sum(
            lane < n_warps ? partial[lane] : T(0), mask);
        if (lane < n_warps) partial[lane] = total;
    }
    __syncthreads();
    return local + (warp ? partial[warp - 1] : T(0));
}

template <class T>
__device__ T cunumpy_block_inclusive_sum(T v)
{ return cunumpy_block_scan_sum<T, false>(v); }

template <class T>
__device__ T cunumpy_block_exclusive_sum(T v)
{ return cunumpy_block_scan_sum<T, true>(v); }

#endif  // CUNUMPY_SCAN_CUH
