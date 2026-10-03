// cunumpy/morton.cuh: Morton (Z-order) keys for kernels.
//
// A Morton key interleaves the bits of the integer cell coordinates of a
// point. Sorting points by their keys orders them along a Z-shaped curve, and
// the points of every quadtree/octree node on the same box form a contiguous
// range of the sorted array. The keys are equal to those of the host function
// cunumpy.morton_keys when the kernel gets the same lower corner and the
// scales of cunumpy.morton_scales(lower, upper, levels):
//
//     #include <cunumpy/morton.cuh>
//
//     extern "C" __global__ void keys2d(const double* pos, unsigned long long* key,
//                                       long long n, double x0, double y0,
//                                       double sx, double sy, int levels) {
//         long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
//         if (i >= n) return;
//         key[i] = cunumpy_morton_key2(pos[2 * i], pos[2 * i + 1],
//                                      x0, y0, sx, sy, levels);
//     }
//
// With `levels` bits per axis a key has ndim * levels bits; axis 0 is the
// lowest bit of every group of ndim bits, and the top group is the child of
// the root. The cell along an axis is floor((x - lower) * scale), clipped to
// [0, 2^levels - 1]: the same operations, in the same order, as on the host,
// so the keys are bit-identical. Positions must be finite.
//
// Plain integer and double arithmetic only, so the header also compiles as
// C++ (see cunumpy.kernel_testing.emulate_cuda_kernel).

#ifndef CUNUMPY_MORTON_CUH
#define CUNUMPY_MORTON_CUH

// Spread the low 32 bits of v to the even bits of a 64-bit word.
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_spread2(
    unsigned long long v)
{
    v &= 0xFFFFFFFFull;
    v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
    v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
    v = (v | (v << 2)) & 0x3333333333333333ull;
    v = (v | (v << 1)) & 0x5555555555555555ull;
    return v;
}

// Inverse of cunumpy_morton_spread2: gather the even bits of v.
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_compact2(
    unsigned long long v)
{
    v &= 0x5555555555555555ull;
    v = (v ^ (v >> 1)) & 0x3333333333333333ull;
    v = (v ^ (v >> 2)) & 0x0F0F0F0F0F0F0F0Full;
    v = (v ^ (v >> 4)) & 0x00FF00FF00FF00FFull;
    v = (v ^ (v >> 8)) & 0x0000FFFF0000FFFFull;
    v = (v ^ (v >> 16)) & 0xFFFFFFFFull;
    return v;
}

// Spread the low 21 bits of v to every third bit of a 64-bit word.
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_spread3(
    unsigned long long v)
{
    v &= 0x1FFFFFull;
    v = (v | (v << 32)) & 0x001F00000000FFFFull;
    v = (v | (v << 16)) & 0x001F0000FF0000FFull;
    v = (v | (v << 8)) & 0x100F00F00F00F00Full;
    v = (v | (v << 4)) & 0x10C30C30C30C30C3ull;
    v = (v | (v << 2)) & 0x1249249249249249ull;
    return v;
}

// Inverse of cunumpy_morton_spread3.
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_compact3(
    unsigned long long v)
{
    v &= 0x1249249249249249ull;
    v = (v ^ (v >> 2)) & 0x10C30C30C30C30C3ull;
    v = (v ^ (v >> 4)) & 0x100F00F00F00F00Full;
    v = (v ^ (v >> 8)) & 0x001F0000FF0000FFull;
    v = (v ^ (v >> 16)) & 0x001F00000000FFFFull;
    v = (v ^ (v >> 32)) & 0x1FFFFFull;
    return v;
}

// Key of integer cells (ix, iy) / (ix, iy, iz), as cunumpy.morton_encode.
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_encode2(
    unsigned long long ix, unsigned long long iy)
{
    return cunumpy_morton_spread2(ix) | (cunumpy_morton_spread2(iy) << 1);
}

__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_encode3(
    unsigned long long ix, unsigned long long iy, unsigned long long iz)
{
    return cunumpy_morton_spread3(ix) | (cunumpy_morton_spread3(iy) << 1)
         | (cunumpy_morton_spread3(iz) << 2);
}

// Cell of x along one axis: floor((x - lower) * scale) in [0, 2^levels - 1].
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_cell(
    double x, double lower, double scale, int levels)
{
    const double top = (double)((1ull << levels) - 1ull);
    double c = floor((x - lower) * scale);
    c = c < 0.0 ? 0.0 : c;
    c = c > top ? top : c;
    return (unsigned long long)c;
}

// Key of a point, as cunumpy.morton_keys with scales = morton_scales(...).
__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_key2(
    double x, double y, double lower_x, double lower_y,
    double scale_x, double scale_y, int levels)
{
    return cunumpy_morton_encode2(cunumpy_morton_cell(x, lower_x, scale_x, levels),
                                  cunumpy_morton_cell(y, lower_y, scale_y, levels));
}

__host__ __device__ __forceinline__ unsigned long long cunumpy_morton_key3(
    double x, double y, double z, double lower_x, double lower_y, double lower_z,
    double scale_x, double scale_y, double scale_z, int levels)
{
    return cunumpy_morton_encode3(cunumpy_morton_cell(x, lower_x, scale_x, levels),
                                  cunumpy_morton_cell(y, lower_y, scale_y, levels),
                                  cunumpy_morton_cell(z, lower_z, scale_z, levels));
}

#endif  // CUNUMPY_MORTON_CUH
