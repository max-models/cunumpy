# Metal kernels on Apple silicon

`MetalKernel` runs a kernel written in the Metal Shading Language (MSL) on the
GPU of an Apple silicon Mac. It uses [MLX](https://github.com/ml-explore/mlx),
Apple's array framework, to compile and launch the kernel, so no Objective-C or
Xcode project is needed. It is the Mac counterpart of
[`CudaKernel`](cuda-kernel.md), with two differences:

* it works on **NumPy arrays**: there is no Metal backend, and the kernel copies
  its arguments to MLX arrays and the results back into your output arrays;
* it is **float32 only**, because the Apple GPU has no float64.

Install it with `pip install 'cunumpy[metal]'` (MLX is installed on arm64 Macs
only). `xp.kernels.metal_available()` is `True` when MLX is installed and a Metal
GPU is present.

## A first kernel

```python
import numpy as np
import cunumpy as xp

scale = xp.kernels.MetalKernel(
    "uint i = thread_position_in_grid.x; y[i] = a[0] * x[i];",
    inputs=["x", "a"],
    outputs=["y"],
)

x = np.arange(8, dtype=np.float32)
y = np.empty_like(x)
scale(x, 2.0, out=y)
```

The essentials:

* `source` is only the **body** of the kernel function. MLX writes the
  signature from `inputs` and `outputs`: each name is a pointer to the flat,
  row-major data of that array, so a 2D array `a` of shape `(n, 3)` is read as
  `a[3 * i + j]`. Attributes such as `thread_position_in_grid` are added to the
  signature when the body uses them.
* Inputs are passed in the order of `inputs`. A Python `float` or `int` becomes
  a one-element `float32` or `int32` array; read it as `a[0]`.
* `out` is one array, or one array per name in `outputs`. Their shapes and dtypes
  define the outputs, and they are filled in place. The call returns them.
* `n_threads` is the **total** number of threads, not the number of threadgroups
  (the opposite of the CUDA `grid`). It defaults to the first axis of the first
  output. Threads beyond the data must return early, as for CUDA, if the thread
  count is not a multiple of the threadgroup size.
* Outputs start uninitialized. Write every element, or pass the old array as an
  input too when the kernel updates it, or give `init_value=0.0`.
* Compilation happens at the first call and MLX caches the result.

## Compile-time constants

`template` gives constants that are known when the kernel is compiled, which
lets the compiler unroll loops. Each name is available in the source:

```python
push = xp.kernels.MetalKernel(
    """
    uint i = thread_position_in_grid.x;
    float x = pos[i], v = vel[i];
    for (int k = 0; k < NSTEPS; ++k) { x += v * dt[0]; }
    pos_out[i] = x;
    """,
    inputs=["pos", "vel", "dt"],
    outputs=["pos_out"],
)
push(pos, vel, 0.01, out=pos_out, template={"NSTEPS": 200})
```

A different value of `NSTEPS` compiles another variant. `header` takes
`#include`s, `#define`s and helper functions that go before the kernel.

## float32 and float64

An Apple GPU computes in float32. A float64 array passed to a `MetalKernel`
raises `TypeError` that names the argument, so no precision is lost silently. If
float32 is accurate enough for the kernel, create it with `float64="cast"`:
float64 inputs are computed in float32 and float64 outputs are filled from the
float32 results. A particle push over 200 steps agreed with a float64 reference
to about 3e-5, but whether that is acceptable depends on the physics, so check it
against the host kernel with
[`assert_kernels_agree`](testing.md) using a float32 tolerance.

## Cost of the copies

Every call copies the inputs to MLX arrays and the results back. The copies are
counted as `to_device` and `to_host` by
[`count_transfers()`](../guides/profiling.md), so they can be found in a profile.
On an M1, 4 million particles (3 float32 coordinates each) took about 5 ms to
copy to the GPU and 2 ms to copy back, next to a push kernel of 11.5 ms. A kernel
that runs once per time step on fresh NumPy arrays therefore pays a noticeable
share of its time in copies, and a kernel with little work per element (an
`a * x + y` update) is no faster than NumPy. The GPU pays off for kernels with
much arithmetic per element, such as pushers and interpolation.

## What it does not do

* It is not part of the [`Kernel`](dispatch.md) dispatch: a `Kernel` pairs a host
  kernel with a CUDA kernel only, so call a `MetalKernel` directly.
* It does not keep data on the GPU between calls.
* It cannot be tested on GitHub's hosted macOS runners, which are virtual
  machines without a Metal GPU. Its launch tests are skipped there and run on a
  Mac with `pytest tests/unit/test_metal_kernel.py -rs`.
