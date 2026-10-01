// Thread-index helpers for CUDA kernels.
//
//     #include <cunumpy/index.cuh>
//
//     extern "C" __global__ void axpy(double a, const double* x, double* y,
//                                     long long n) {
//         CUNUMPY_THREAD_1D(i, n);  // long long i; returns if i >= n
//         y[i] += a * x[i];
//     }
//
// Indices are `long long`, so that they can address more than 2^31 elements
// and match the `shape`/`strides` of the views in array_view.cuh.

#ifndef CUNUMPY_INDEX_CUH
#define CUNUMPY_INDEX_CUH

// Global thread index along x (y, z); one value per thread.
#define CUNUMPY_GLOBAL_INDEX_X() \
    ((long long)blockDim.x * blockIdx.x + threadIdx.x)
#define CUNUMPY_GLOBAL_INDEX_Y() \
    ((long long)blockDim.y * blockIdx.y + threadIdx.y)
#define CUNUMPY_GLOBAL_INDEX_Z() \
    ((long long)blockDim.z * blockIdx.z + threadIdx.z)

// Declares `long long i` as the global thread index and returns from the
// kernel if `i >= n`. Use with `n_threads=n` in Python.
#define CUNUMPY_THREAD_1D(i, n)             \
    long long i = CUNUMPY_GLOBAL_INDEX_X(); \
    if (i >= (long long)(n)) return

// Likewise in 2 and 3 dimensions, for `n_threads=(ni, nj)` / `(ni, nj, nk)`.
#define CUNUMPY_THREAD_2D(i, j, ni, nj)     \
    long long i = CUNUMPY_GLOBAL_INDEX_X(); \
    long long j = CUNUMPY_GLOBAL_INDEX_Y(); \
    if (i >= (long long)(ni) || j >= (long long)(nj)) return

#define CUNUMPY_THREAD_3D(i, j, k, ni, nj, nk)                  \
    long long i = CUNUMPY_GLOBAL_INDEX_X();                     \
    long long j = CUNUMPY_GLOBAL_INDEX_Y();                     \
    long long k = CUNUMPY_GLOBAL_INDEX_Z();                     \
    if (i >= (long long)(ni) || j >= (long long)(nj) ||         \
        k >= (long long)(nk)) return

// Grid-stride loop over `i` in [0, n): every thread handles several elements,
// so any launch size works (e.g. `grid=` a fixed number of blocks in Python):
//
//     CUNUMPY_GRID_STRIDE_1D(i, n) { y[i] += a * x[i]; }
#define CUNUMPY_GRID_STRIDE_1D(i, n)                                 \
    for (long long i = CUNUMPY_GLOBAL_INDEX_X(); i < (long long)(n); \
         i += (long long)blockDim.x * gridDim.x)

#endif  // CUNUMPY_INDEX_CUH
