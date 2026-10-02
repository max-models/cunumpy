// Strided array views for CUDA kernels, passed by value from Python.
//
// Array1D<T> to Array4D<T> describe a (possibly non-contiguous)
// device array the way NumPy/CuPy do: a data pointer, a shape and strides.
// Strides are in ELEMENTS, not bytes, so that `a(i, j)` is
// `data[i * strides[0] + j * strides[1]]`. Elements are accessed with
// `operator()`, which mirrors the `a[i, j]` indexing of the pyccel kernels
// being ported.
//
// The views are created on the Python side by cunumpy (a kernel parameter or a
// CudaStruct field of type `Array2D<double>` takes a CuPy array). They are
// passed by value, so the memory layout must be exactly, for ndim = 1 to 4:
//
//     T* data;                  // 8 bytes
//     long long shape[ndim];    // ndim * 8 bytes
//     long long strides[ndim];  // ndim * 8 bytes
//
// with 8-byte alignment and no padding (sizeof == 8 * (1 + 2 * ndim)). Do not
// add data members, virtual functions or a base class; the static_asserts at
// the end of this file check the size. Member functions do not change the
// layout.
//
// Bounds checks: compile with -DCUNUMPY_BOUNDS_CHECK to check every index
// against the shape (an out-of-bounds index prints a message and traps the
// kernel, which CuPy reports as a CUDA error). Without the macro, indexing is
// unchecked.

#ifndef CUNUMPY_ARRAY_VIEW_CUH
#define CUNUMPY_ARRAY_VIEW_CUH

#ifdef CUNUMPY_BOUNDS_CHECK
#define CUNUMPY_CHECK_INDEX(index, axis, extent)                                 \
    do {                                                                         \
        if ((index) < 0 || (index) >= (extent)) {                                \
            printf("cunumpy: index %lld is out of bounds for axis %d with size " \
                   "%lld\n",                                                     \
                   (long long)(index), (int)(axis), (long long)(extent));        \
            __trap();                                                            \
        }                                                                        \
    } while (0)
#else
#define CUNUMPY_CHECK_INDEX(index, axis, extent) ((void)0)
#endif

template <typename T>
struct Array1D {
    T* data;
    long long shape[1];
    long long strides[1];

    __device__ __forceinline__ T& operator()(long long i) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        return data[i * strides[0]];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const { return shape[0]; }
};

template <typename T>
struct Array2D {
    T* data;
    long long shape[2];
    long long strides[2];

    __device__ __forceinline__ T& operator()(long long i, long long j) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        return data[i * strides[0] + j * strides[1]];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1];
    }
};

template <typename T>
struct Array3D {
    T* data;
    long long shape[3];
    long long strides[3];

    __device__ __forceinline__ T& operator()(long long i, long long j,
                                             long long k) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        CUNUMPY_CHECK_INDEX(k, 2, shape[2]);
        return data[i * strides[0] + j * strides[1] + k * strides[2]];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1] * shape[2];
    }
};

// A 4D view, e.g. a 3D grid of vector components (nx, ny, nz, ncomp).
template <typename T>
struct Array4D {
    T* data;
    long long shape[4];
    long long strides[4];

    __device__ __forceinline__ T& operator()(long long i, long long j,
                                             long long k, long long l) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        CUNUMPY_CHECK_INDEX(k, 2, shape[2]);
        CUNUMPY_CHECK_INDEX(l, 3, shape[3]);
        return data[i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3]];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }
};

// The layout the Python side packs: pointer, shape, strides, 8-byte aligned.
static_assert(sizeof(Array1D<double>) == 24, "unexpected Array1D layout");
static_assert(sizeof(Array2D<double>) == 40, "unexpected Array2D layout");
static_assert(sizeof(Array3D<double>) == 56, "unexpected Array3D layout");
static_assert(sizeof(Array4D<double>) == 72, "unexpected Array4D layout");
static_assert(sizeof(Array1D<char>) == 24, "unexpected Array1D layout");
static_assert(alignof(Array2D<float>) == 8, "unexpected Array2D alignment");

#endif  // CUNUMPY_ARRAY_VIEW_CUH
