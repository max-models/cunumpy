// Array views for CUDA kernels, passed by value from Python.
//
// Array1D<T> to Array16D<T> describe a (possibly non-contiguous)
// device array the way NumPy/CuPy do: a data pointer, a shape and strides.
// Strides are in ELEMENTS, not bytes, so that `a(i, j)` is
// `data[i * strides[0] + j * strides[1]]`. Elements are accessed with
// `operator()`, which mirrors the `a[i, j]` indexing of the pyccel kernels
// being ported.
//
// The views are created on the Python side by cunumpy (a kernel parameter or a
// CudaStruct field of type `Array2D<double>` takes a CuPy array). They are
// passed by value, so the memory layout must be exactly, for ndim = 1 to 16:
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
// CArray1D<T> to CArray16D<T> are C-contiguous (row-major) views: a data
// pointer and a shape, no strides. `a(i, j)` is `data[i * shape[1] + j]`, so
// the last index is always the fast one and the compiler knows it has unit
// stride. A parameter or field of these types only takes C-contiguous arrays;
// cunumpy raises for a non-contiguous view instead of copying it (a copy would
// silently drop what the kernel writes). Their layout, for ndim = 1 to 16:
//
//     T* data;                  // 8 bytes
//     long long shape[ndim];    // ndim * 8 bytes
//
// (sizeof == 8 * (1 + ndim)). A CArrayND<T> converts to an ArrayND<T>, so it
// can be passed to device functions written for strided views.
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

// C-contiguous views: shape only, the strides follow from it.

template <typename T>
struct CArray1D {
    T* data;
    long long shape[1];

    __device__ __forceinline__ T& operator()(long long i) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        return data[i];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const { return shape[0]; }

    __host__ __device__ operator Array1D<T>() const {
        return Array1D<T>{data, {shape[0]}, {1}};
    }
};

template <typename T>
struct CArray2D {
    T* data;
    long long shape[2];

    __device__ __forceinline__ T& operator()(long long i, long long j) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        return data[i * shape[1] + j];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1];
    }

    __host__ __device__ operator Array2D<T>() const {
        return Array2D<T>{data, {shape[0], shape[1]}, {shape[1], 1}};
    }
};

template <typename T>
struct CArray3D {
    T* data;
    long long shape[3];

    __device__ __forceinline__ T& operator()(long long i, long long j,
                                             long long k) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        CUNUMPY_CHECK_INDEX(k, 2, shape[2]);
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1] * shape[2];
    }

    __host__ __device__ operator Array3D<T>() const {
        return Array3D<T>{data,
                          {shape[0], shape[1], shape[2]},
                          {shape[1] * shape[2], shape[2], 1}};
    }
};

template <typename T>
struct CArray4D {
    T* data;
    long long shape[4];

    __device__ __forceinline__ T& operator()(long long i, long long j,
                                             long long k, long long l) const {
        CUNUMPY_CHECK_INDEX(i, 0, shape[0]);
        CUNUMPY_CHECK_INDEX(j, 1, shape[1]);
        CUNUMPY_CHECK_INDEX(k, 2, shape[2]);
        CUNUMPY_CHECK_INDEX(l, 3, shape[3]);
        return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
    }

    // Number of elements.
    __device__ __forceinline__ long long size() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }

    __host__ __device__ operator Array4D<T>() const {
        return Array4D<T>{data,
                          {shape[0], shape[1], shape[2], shape[3]},
                          {shape[1] * shape[2] * shape[3], shape[2] * shape[3],
                           shape[3], 1}};
    }
};

// Higher-dimensional views share an implementation; aliases keep the named
// types used by kernel signatures and Python's argument packer.
namespace cunumpy_detail {
template <typename T, int N>
struct ArrayView {
    T* data;
    long long shape[N];
    long long strides[N];

    template <typename... Indices>
    __device__ __forceinline__ T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == N, "wrong number of array indices");
        const long long index[N] = {static_cast<long long>(indices)...};
        long long offset = 0;
        #pragma unroll
        for (int axis = 0; axis < N; ++axis) {
            CUNUMPY_CHECK_INDEX(index[axis], axis, shape[axis]);
            offset += index[axis] * strides[axis];
        }
        return data[offset];
    }

    __device__ __forceinline__ long long size() const {
        long long result = 1;
        #pragma unroll
        for (int axis = 0; axis < N; ++axis) result *= shape[axis];
        return result;
    }
};

template <typename T, int N>
struct CArrayView {
    T* data;
    long long shape[N];

    template <typename... Indices>
    __device__ __forceinline__ T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == N, "wrong number of array indices");
        const long long index[N] = {static_cast<long long>(indices)...};
        long long offset = 0;
        #pragma unroll
        for (int axis = 0; axis < N; ++axis) {
            CUNUMPY_CHECK_INDEX(index[axis], axis, shape[axis]);
            offset = offset * shape[axis] + index[axis];
        }
        return data[offset];
    }

    __device__ __forceinline__ long long size() const {
        long long result = 1;
        #pragma unroll
        for (int axis = 0; axis < N; ++axis) result *= shape[axis];
        return result;
    }

    __host__ __device__ operator ArrayView<T, N>() const {
        ArrayView<T, N> result{};
        result.data = data;
        long long stride = 1;
        #pragma unroll
        for (int axis = N - 1; axis >= 0; --axis) {
            result.shape[axis] = shape[axis];
            result.strides[axis] = stride;
            stride *= shape[axis];
        }
        return result;
    }
};
}  // namespace cunumpy_detail

template <typename T> using Array5D = cunumpy_detail::ArrayView<T, 5>;
template <typename T> using CArray5D = cunumpy_detail::CArrayView<T, 5>;
template <typename T> using Array6D = cunumpy_detail::ArrayView<T, 6>;
template <typename T> using CArray6D = cunumpy_detail::CArrayView<T, 6>;
template <typename T> using Array7D = cunumpy_detail::ArrayView<T, 7>;
template <typename T> using CArray7D = cunumpy_detail::CArrayView<T, 7>;
template <typename T> using Array8D = cunumpy_detail::ArrayView<T, 8>;
template <typename T> using CArray8D = cunumpy_detail::CArrayView<T, 8>;
template <typename T> using Array9D = cunumpy_detail::ArrayView<T, 9>;
template <typename T> using CArray9D = cunumpy_detail::CArrayView<T, 9>;
template <typename T> using Array10D = cunumpy_detail::ArrayView<T, 10>;
template <typename T> using CArray10D = cunumpy_detail::CArrayView<T, 10>;
template <typename T> using Array11D = cunumpy_detail::ArrayView<T, 11>;
template <typename T> using CArray11D = cunumpy_detail::CArrayView<T, 11>;
template <typename T> using Array12D = cunumpy_detail::ArrayView<T, 12>;
template <typename T> using CArray12D = cunumpy_detail::CArrayView<T, 12>;
template <typename T> using Array13D = cunumpy_detail::ArrayView<T, 13>;
template <typename T> using CArray13D = cunumpy_detail::CArrayView<T, 13>;
template <typename T> using Array14D = cunumpy_detail::ArrayView<T, 14>;
template <typename T> using CArray14D = cunumpy_detail::CArrayView<T, 14>;
template <typename T> using Array15D = cunumpy_detail::ArrayView<T, 15>;
template <typename T> using CArray15D = cunumpy_detail::CArrayView<T, 15>;
template <typename T> using Array16D = cunumpy_detail::ArrayView<T, 16>;
template <typename T> using CArray16D = cunumpy_detail::CArrayView<T, 16>;

// The layouts the Python side packs: pointer, shape (and strides), 8-byte
// aligned.
static_assert(sizeof(Array1D<double>) == 24, "unexpected Array1D layout");
static_assert(sizeof(Array2D<double>) == 40, "unexpected Array2D layout");
static_assert(sizeof(Array3D<double>) == 56, "unexpected Array3D layout");
static_assert(sizeof(Array4D<double>) == 72, "unexpected Array4D layout");
static_assert(sizeof(Array1D<char>) == 24, "unexpected Array1D layout");
static_assert(alignof(Array2D<float>) == 8, "unexpected Array2D alignment");
static_assert(sizeof(CArray1D<double>) == 16, "unexpected CArray1D layout");
static_assert(sizeof(CArray2D<double>) == 24, "unexpected CArray2D layout");
static_assert(sizeof(CArray3D<double>) == 32, "unexpected CArray3D layout");
static_assert(sizeof(CArray4D<double>) == 40, "unexpected CArray4D layout");
static_assert(alignof(CArray2D<float>) == 8, "unexpected CArray2D alignment");

static_assert(sizeof(Array5D<double>) == 88, "unexpected Array5D layout");
static_assert(alignof(Array5D<float>) == 8, "unexpected Array5D alignment");
static_assert(sizeof(CArray5D<double>) == 48, "unexpected CArray5D layout");
static_assert(alignof(CArray5D<float>) == 8, "unexpected CArray5D alignment");
static_assert(sizeof(Array6D<double>) == 104, "unexpected Array6D layout");
static_assert(alignof(Array6D<float>) == 8, "unexpected Array6D alignment");
static_assert(sizeof(CArray6D<double>) == 56, "unexpected CArray6D layout");
static_assert(alignof(CArray6D<float>) == 8, "unexpected CArray6D alignment");
static_assert(sizeof(Array7D<double>) == 120, "unexpected Array7D layout");
static_assert(alignof(Array7D<float>) == 8, "unexpected Array7D alignment");
static_assert(sizeof(CArray7D<double>) == 64, "unexpected CArray7D layout");
static_assert(alignof(CArray7D<float>) == 8, "unexpected CArray7D alignment");
static_assert(sizeof(Array8D<double>) == 136, "unexpected Array8D layout");
static_assert(alignof(Array8D<float>) == 8, "unexpected Array8D alignment");
static_assert(sizeof(CArray8D<double>) == 72, "unexpected CArray8D layout");
static_assert(alignof(CArray8D<float>) == 8, "unexpected CArray8D alignment");
static_assert(sizeof(Array9D<double>) == 152, "unexpected Array9D layout");
static_assert(alignof(Array9D<float>) == 8, "unexpected Array9D alignment");
static_assert(sizeof(CArray9D<double>) == 80, "unexpected CArray9D layout");
static_assert(alignof(CArray9D<float>) == 8, "unexpected CArray9D alignment");
static_assert(sizeof(Array10D<double>) == 168, "unexpected Array10D layout");
static_assert(alignof(Array10D<float>) == 8, "unexpected Array10D alignment");
static_assert(sizeof(CArray10D<double>) == 88, "unexpected CArray10D layout");
static_assert(alignof(CArray10D<float>) == 8, "unexpected CArray10D alignment");
static_assert(sizeof(Array11D<double>) == 184, "unexpected Array11D layout");
static_assert(alignof(Array11D<float>) == 8, "unexpected Array11D alignment");
static_assert(sizeof(CArray11D<double>) == 96, "unexpected CArray11D layout");
static_assert(alignof(CArray11D<float>) == 8, "unexpected CArray11D alignment");
static_assert(sizeof(Array12D<double>) == 200, "unexpected Array12D layout");
static_assert(alignof(Array12D<float>) == 8, "unexpected Array12D alignment");
static_assert(sizeof(CArray12D<double>) == 104, "unexpected CArray12D layout");
static_assert(alignof(CArray12D<float>) == 8, "unexpected CArray12D alignment");
static_assert(sizeof(Array13D<double>) == 216, "unexpected Array13D layout");
static_assert(alignof(Array13D<float>) == 8, "unexpected Array13D alignment");
static_assert(sizeof(CArray13D<double>) == 112, "unexpected CArray13D layout");
static_assert(alignof(CArray13D<float>) == 8, "unexpected CArray13D alignment");
static_assert(sizeof(Array14D<double>) == 232, "unexpected Array14D layout");
static_assert(alignof(Array14D<float>) == 8, "unexpected Array14D alignment");
static_assert(sizeof(CArray14D<double>) == 120, "unexpected CArray14D layout");
static_assert(alignof(CArray14D<float>) == 8, "unexpected CArray14D alignment");
static_assert(sizeof(Array15D<double>) == 248, "unexpected Array15D layout");
static_assert(alignof(Array15D<float>) == 8, "unexpected Array15D alignment");
static_assert(sizeof(CArray15D<double>) == 128, "unexpected CArray15D layout");
static_assert(alignof(CArray15D<float>) == 8, "unexpected CArray15D alignment");
static_assert(sizeof(Array16D<double>) == 264, "unexpected Array16D layout");
static_assert(alignof(Array16D<float>) == 8, "unexpected Array16D alignment");
static_assert(sizeof(CArray16D<double>) == 136, "unexpected CArray16D layout");
static_assert(alignof(CArray16D<float>) == 8, "unexpected CArray16D alignment");

#endif  // CUNUMPY_ARRAY_VIEW_CUH
