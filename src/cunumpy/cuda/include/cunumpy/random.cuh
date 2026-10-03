// cunumpy/random.cuh: counter-based random numbers (Philox4x32-10) for kernels.
//
// A counter-based generator has no state: the numbers are a pure function of
// a key (the seed) and a counter. Each particle draws its own numbers from
// (seed, stream = particle id, counter = draw index), so a kernel needs no
// per-thread generator state, the result does not depend on the launch shape
// or the order of the threads, and the host computes exactly the same numbers
// (cunumpy.philox_uniform / philox_normal), so a kernel that samples random
// numbers can be compared with its host version.
//
//     #include <cunumpy/random.cuh>
//
//     extern "C" __global__ void thermalize(double* v, long long n,
//                                           unsigned long long seed,
//                                           unsigned long long step, double v_th) {
//         long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
//         if (i >= n) return;
//         double z0, z1;
//         cunumpy_normal2(seed, (unsigned long long)i, step, &z0, &z1);
//         v[i] = v_th * z0;
//     }
//
// Philox4x32-10 is the generator of Salmon et al., "Parallel random numbers:
// as easy as 1, 2, 3" (SC'11), as in Random123 and cuRAND. One call maps a
// 128-bit counter and a 64-bit key to 128 random bits:
//     counter = (counter low, counter high, stream low, stream high), key = seed.
// The uniform numbers are bit-identical on host and device. The normal numbers
// (Box-Muller: log, sqrt, sin, cos) can differ in the last bits, since the GPU's
// math functions are not the host's.
//
// Everything is plain integer arithmetic (no CUDA vector types or intrinsics),
// so the header also compiles as C++ (see cunumpy.testing.emulate_cuda_kernel).

#ifndef CUNUMPY_RANDOM_CUH
#define CUNUMPY_RANDOM_CUH

struct cunumpy_u32x4 {
    unsigned int v[4];
};

#define CUNUMPY_PHILOX_M0 0xD2511F53u
#define CUNUMPY_PHILOX_M1 0xCD9E8D57u
#define CUNUMPY_PHILOX_W0 0x9E3779B9u
#define CUNUMPY_PHILOX_W1 0xBB67AE85u

// Philox4x32-10: 128 random bits from a 128-bit counter and a 64-bit key.
__device__ __forceinline__ cunumpy_u32x4 cunumpy_philox4x32_10(
    cunumpy_u32x4 ctr, unsigned int key0, unsigned int key1)
{
    for (int round = 0; round < 10; ++round) {
        const unsigned long long p0 = (unsigned long long)CUNUMPY_PHILOX_M0 * ctr.v[0];
        const unsigned long long p1 = (unsigned long long)CUNUMPY_PHILOX_M1 * ctr.v[2];
        const unsigned int hi0 = (unsigned int)(p0 >> 32), lo0 = (unsigned int)p0;
        const unsigned int hi1 = (unsigned int)(p1 >> 32), lo1 = (unsigned int)p1;
        cunumpy_u32x4 next;
        next.v[0] = hi1 ^ ctr.v[1] ^ key0;
        next.v[1] = lo1;
        next.v[2] = hi0 ^ ctr.v[3] ^ key1;
        next.v[3] = lo0;
        ctr = next;
        key0 += CUNUMPY_PHILOX_W0;
        key1 += CUNUMPY_PHILOX_W1;
    }
    return ctr;
}

// The 128 random bits for (seed, stream, counter).
__device__ __forceinline__ cunumpy_u32x4 cunumpy_random_bits(
    unsigned long long seed, unsigned long long stream, unsigned long long counter)
{
    cunumpy_u32x4 ctr;
    ctr.v[0] = (unsigned int)counter;
    ctr.v[1] = (unsigned int)(counter >> 32);
    ctr.v[2] = (unsigned int)stream;
    ctr.v[3] = (unsigned int)(stream >> 32);
    return cunumpy_philox4x32_10(ctr, (unsigned int)seed, (unsigned int)(seed >> 32));
}

// A double in [0, 1) with 53 random bits, from two 32-bit words.
__device__ __forceinline__ double cunumpy_u64_to_uniform(unsigned int hi, unsigned int lo)
{
    const unsigned long long bits = ((unsigned long long)hi << 32) | lo;
    return (double)(bits >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

// Two uniform doubles in [0, 1) for (seed, stream, counter).
__device__ __forceinline__ void cunumpy_uniform2(
    unsigned long long seed, unsigned long long stream, unsigned long long counter,
    double* u0, double* u1)
{
    const cunumpy_u32x4 r = cunumpy_random_bits(seed, stream, counter);
    *u0 = cunumpy_u64_to_uniform(r.v[0], r.v[1]);
    *u1 = cunumpy_u64_to_uniform(r.v[2], r.v[3]);
}

// One uniform double in [0, 1): the first of cunumpy_uniform2.
__device__ __forceinline__ double cunumpy_uniform(
    unsigned long long seed, unsigned long long stream, unsigned long long counter)
{
    const cunumpy_u32x4 r = cunumpy_random_bits(seed, stream, counter);
    return cunumpy_u64_to_uniform(r.v[0], r.v[1]);
}

// Two independent standard normal doubles for (seed, stream, counter), by the
// Box-Muller transform of cunumpy_uniform2 (1 - u0 is in (0, 1], so log is finite).
__device__ __forceinline__ void cunumpy_normal2(
    unsigned long long seed, unsigned long long stream, unsigned long long counter,
    double* z0, double* z1)
{
    double u0, u1;
    cunumpy_uniform2(seed, stream, counter, &u0, &u1);
    const double radius = sqrt(-2.0 * log(1.0 - u0));
    const double angle = 6.283185307179586 * u1;  // 2 pi
    *z0 = radius * cos(angle);
    *z1 = radius * sin(angle);
}

// One standard normal double: the first of cunumpy_normal2.
__device__ __forceinline__ double cunumpy_normal(
    unsigned long long seed, unsigned long long stream, unsigned long long counter)
{
    double z0, z1;
    cunumpy_normal2(seed, stream, counter, &z0, &z1);
    return z0;
}

#endif  // CUNUMPY_RANDOM_CUH
