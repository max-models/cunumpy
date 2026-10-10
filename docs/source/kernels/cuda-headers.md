# CUDA headers

CuNumpy ships CUDA headers for hand-written kernels. Every `CudaKernel` finds
them without configuration: it adds their directory to its NVRTC options as
`-I<dir>` (once), so a kernel source can write `#include <cunumpy/atomic.cuh>`.
`xp.cuda.cuda_include_dir()` returns that directory (a `str`) to pass the same
headers to other compilers. The shipped headers take part in the compile cache
key also when included in angle brackets, so upgrading CuNumpy with a changed
header recompiles the kernels that use it (see "Headers and the compile cache"
in [Writing CUDA kernels](cuda-kernel.md)).

| Header | Provides |
| --- | --- |
| `cunumpy/index.cuh` | thread-index macros |
| `cunumpy/array_view.cuh` | strided views `Array1D<T>` to `Array16D<T>` |
| `cunumpy/atomic.cuh` | atomic adds, also into 2D/3D arrays |
| `cunumpy/reduce.cuh` | warp and block reductions |
| `cunumpy/scan.cuh` | warp and block prefix sums |
| `cunumpy/random.cuh` | counter-based random numbers, equal to `xp.rng.philox_*` |
| `cunumpy/morton.cuh` | Morton (Z-order) keys, equal to `xp.algorithms.morton_*` |

**AMD/ROCm.** CuPy's own API is the same on a ROCm build, so `xp.set_backend("cupy")`
and the device/memory/stream helpers of `xp.cuda` work unchanged; check
`xp.cuda.is_hip()` to tell the two apart where it matters. `CudaKernel`
compiles plain CUDA C through `cupy.RawKernel`, which HIPRTC accepts for
most kernels. Three known gaps, found on real HIP/ROCm hardware (an AMD
Instinct MI210) and all raising `NotImplementedError` at
`CudaKernel.compile()` on HIP rather than silently misbehaving, none yet
root-caused:

* `cunumpy/reduce.cuh` (and `cunumpy/scan.cuh`, which includes it), see below.
* A kernel taking an `Array5D`/`CArray5D` view or higher (`Array1D` to
  `Array4D` are unaffected): launching reliably corrupts the device, even for
  an in-bounds access (a HIP code-generation or by-value struct-argument
  issue in the generic, variadic-template `ArrayView<T, N>` is suspected).
* A kernel that includes `cunumpy/array_view.cuh` and compiles with
  `CUNUMPY_BOUNDS_CHECK` (directly, or via debug mode, which adds it): merely
  compiling the resulting `printf()` and trap reliably corrupts the device on
  any launch, even one that never takes an out-of-bounds index (a HIP
  device-`printf` or trap-instruction issue is suspected).

Use `cunumpy.kernel_testing`'s `requires_warp_shuffle`, `requires_high_dim_cuda_views`
and `requires_bounds_check_views` markers (or their `*_available()` functions)
to skip tests that hit these on HIP, the way cunumpy's own test suite does.

## `cunumpy/index.cuh`

```c
#include <cunumpy/index.cuh>

CUNUMPY_THREAD_1D(i, n);                 // long long i = global x index; return if i >= n
CUNUMPY_THREAD_2D(i, j, ni, nj);         // 2D launch, n_threads=(ni, nj)
CUNUMPY_THREAD_3D(i, j, k, ni, nj, nk);
CUNUMPY_GRID_STRIDE_1D(i, n) { ... }     // grid-stride loop; launch with any grid
CUNUMPY_GLOBAL_INDEX_X()                 // also _Y(), _Z(): the global index as long long
```

The `CUNUMPY_THREAD_*` macros return early, so do not use them before a block
collective of `reduce.cuh` or `scan.cuh`, which every thread of the block must
reach.

## `cunumpy/array_view.cuh`

Strided views `Array1D<T>` to `Array16D<T>`: `T* data`, `long long
shape[ndim]`, `long long strides[ndim]` (in elements, not bytes),
`operator()(i, j, ...)` returning a reference to the element, and `size()`. A
kernel indexes `a(i, j)` like the Pyccel kernel it is ported from indexes
`a[i, j]`, without hand-passed sizes. A 4D view can, for example, describe a 3D
grid of vector components `(nx, ny, nz, ncomp)`. `CArray1D<T>` to
`CArray16D<T>` are C-contiguous views without strides; they convert to the
strided view of the same dimension.

A kernel parameter or a `CudaStruct` field of type `ArrayND<T>`, for the
scalar C types `CudaKernel` supports, takes a CuPy array of that dtype and
number of dimensions (a mismatch raises `TypeError`), contiguous or not. It is
packed by value into pointer, shape and strides with the memory layout of the
C struct: 8-byte aligned, `sizeof == 8 * (1 + 2 * ndim)` (`8 * (1 + ndim)` for
a `CArrayND<T>`); the header checks this with `static_assert`.
`CudaParameter.view_ndim` is the number of dimensions of such a parameter.

Compiling with `options=("-DCUNUMPY_BOUNDS_CHECK",)`, or in [debug
mode](debugging.md), checks every index against the shape: an out-of-bounds
index prints a message and traps the kernel. Without the macro, indexing is
unchecked.

## `cunumpy/atomic.cuh`

For the many-threads-to-one-cell writes of accumulation kernels (see
[Accumulation kernels](accumulation.md)):

```c
#include <cunumpy/atomic.cuh>

double cunumpy_atomic_add(double* p, double v);   // *p += v, returns old *p
float  cunumpy_atomic_add(float* p, float v);
double cunumpy_atomic_add_2d(double* data, long long n1,
                             long long i, long long j, double v);
double cunumpy_atomic_add_3d(double* data, long long n1, long long n2,
                             long long i, long long j, long long k, double v);
```

The scalar and indexed helpers also support `int`, `unsigned int`, `long long`
and `unsigned long long`, returning the old value. Signed 64-bit addition uses
CUDA's unsigned 64-bit atomic with modulo-2^64 arithmetic. The indexed helpers
address C-contiguous arrays of shape `(n0, n1)` and `(n0, n1, n2)`. They wrap
`atomicAdd`, a hardware instruction for `double` from compute capability 6.0
(sm_60) on; older devices use a compare-and-swap loop.

## `cunumpy/reduce.cuh`

Warp- and block-level reductions for hand-written kernels: in-kernel
diagnostics (energy, momentum, total charge, the maximum velocity for a CFL
check) and combining values in a block before one atomic write.

```c
#include <cunumpy/reduce.cuh>

T cunumpy_warp_sum(T v, unsigned mask = 0xffffffffu); // also _min, _max
T cunumpy_block_sum(T v);  T cunumpy_block_min(T v);  T cunumpy_block_max(T v);
void cunumpy_block_sum_to(T* out, T v);  // *out += block sum, one atomic per block
int cunumpy_block_thread();   // linear thread index in a 1D-3D block
int cunumpy_block_threads();  // threads per block
```

`T` is `int`, `unsigned`, `long long`, `unsigned long long`, `float` or
`double`; `block_sum_to` supports the same types. Every participating thread
gets the result.

* Every thread of the block calls the block functions: no early `return`
  (and no `CUNUMPY_THREAD_1D` before them); threads without a value pass the
  identity, e.g. `0.0` for a sum.
* Block functions handle partial warps and any legal block shape, with x
  fastest. They use 32 values of static shared memory per type and may be
  called several times in a kernel.
* Warp functions take an optional nonzero lane mask, also a sparse one: only
  the named lanes participate, and all of them call with the same mask. The
  default requires all 32 lanes. These rules follow [NVIDIA's warp intrinsic
  constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

**Not ported to HIP/ROCm.** This header hardcodes a 32-lane warp and CUDA's
`_sync` shuffle intrinsics; AMD wavefronts are commonly 64 lanes wide (CDNA:
MI100/MI200/MI300), and HIP's shuffles have no mask argument. Rather than
compile silently-wrong reductions, `CudaKernel.compile()` raises
`NotImplementedError` for a kernel that includes `cunumpy/reduce.cuh` (or
`cunumpy/scan.cuh`, which includes it) when the active CuPy build targets
HIP (`xp.cuda.is_hip()` is True).

```c
extern "C" __global__ void kinetic_energy(const double* v, long long n,
                                          double mass, double* energy) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    double e = i < n ? 0.5 * mass * v[i] * v[i] : 0.0;
    cunumpy_block_sum_to(energy, e);  // zero *energy before the launch
}
```

## `cunumpy/scan.cuh`

Warp and block prefix sums, for compaction and binning:

```c
#include <cunumpy/scan.cuh>

T cunumpy_warp_inclusive_sum(T v, unsigned mask = 0xffffffffu);
T cunumpy_warp_exclusive_sum(T v, unsigned mask = 0xffffffffu);
T cunumpy_block_inclusive_sum(T v);
T cunumpy_block_exclusive_sum(T v);
```

The types and participation rules are those of `reduce.cuh`. Warp scans follow
increasing lane IDs of the participating lanes, also for sparse masks; block
scans follow the linear thread index (x fastest), with partial warps
supported. Exclusive scans start at zero. Block scans use 32 shared values per
type and specialization and may be called repeatedly. They are block-local: a
global compaction needs another pass that combines the block totals (see the
example in [Execution helpers](../guides/execution-helpers.md)).

## `cunumpy/random.cuh`

Counter-based random numbers (Philox4x32-10, as in Random123 and cuRAND): a
pure function of a key and a counter, with no generator state. Each thread
draws from `(seed, stream, counter)`, e.g. `(seed, particle id, step)`, and the
host computes the same numbers:

```c
#include <cunumpy/random.cuh>

cunumpy_u32x4 cunumpy_philox4x32_10(cunumpy_u32x4 ctr, unsigned int key0, unsigned int key1);
double cunumpy_uniform(seed, stream, counter);              // [0, 1), 53 bits
void   cunumpy_uniform2(seed, stream, counter, &u0, &u1);  // two from one call
double cunumpy_normal(seed, stream, counter);               // Box-Muller
void   cunumpy_normal2(seed, stream, counter, &z0, &z1);
```

`seed`, `stream` and `counter` are `unsigned long long`. On the host:

```python
ids = xp.arange(n, dtype=xp.uint64)
u0, u1 = xp.rng.philox_uniform2(seed, ids, step)  # == cunumpy_uniform2 in thread i
z0, z1 = xp.rng.philox_normal2(seed, ids, step)
words = xp.rng.philox4x32_10(counter_words, key0, key1)  # the raw generator
```

`xp.rng.philox_uniform`, `philox_uniform2`, `philox_normal`, `philox_normal2`
and `philox4x32_10` broadcast their arguments and return NumPy or CuPy arrays,
matching the inputs. The uniform numbers equal the kernel's bit for bit; the
normal numbers can differ in the last bits (`log`, `sqrt`, `sin` and `cos` on
the GPU are not the host's). The generator passes the Random123 known-answer
tests. Use a different `counter` for every random decision of a step.

## `cunumpy/morton.cuh`

Morton (Z-order) keys: the bits of a point's integer cell coordinates,
interleaved into one `uint64`. Sorted by key, nearby points are nearby in
memory, and the points of every node of a quadtree (2D) or octree (3D) on the
same box form a contiguous range, the starting point of tree builds on the
GPU. On the host:

```python
keys = xp.algorithms.morton_keys(
    positions, lower, upper, levels
)  # (n, 2|3) -> (n,) uint64
keys, order, positions = xp.algorithms.sort_by_key(keys, positions)
node = keys >> np.uint64(ndim * (levels - level))  # node index at `level`
cells = xp.algorithms.morton_decode(node, ndim)  # its integer coordinates
key = xp.algorithms.morton_encode(ix, iy)  # from integer cells
scales = xp.algorithms.morton_scales(
    lower, upper, levels
)  # 2**levels / (upper - lower)
```

In a kernel:

```c
#include <cunumpy/morton.cuh>

unsigned long long cunumpy_morton_key2(x, y, lower_x, lower_y, scale_x, scale_y, levels);
unsigned long long cunumpy_morton_key3(x, y, z, lower_x, ..., scale_x, ..., levels);
unsigned long long cunumpy_morton_encode2(ix, iy);   // and _encode3(ix, iy, iz)
unsigned long long cunumpy_morton_cell(x, lower, scale, levels);
unsigned long long cunumpy_morton_spread2(v);       // and _compact2, _spread3, _compact3
```

`levels` is the number of bits per axis, at most 32 in 2D and 21 in 3D
(`xp.algorithms.MAX_MORTON_LEVELS`). Axis 0 is the lowest bit of every group of
`ndim` bits; the top group is the child of the root. The cell along an axis is
`floor((x - lower) * scale)` clipped to `[0, 2**levels - 1]`: points on a cell
boundary go to the upper cell, points outside the box to the nearest face, and
`lower > upper` reverses the axis. Given the `morton_scales` of the host, the
kernel functions return the host keys bit for bit. All host functions run on
NumPy and CuPy arrays.
