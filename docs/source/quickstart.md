# Quickstart

This page is a ten-minute tour. Each section ends with a link to the guide that
covers the topic in depth.

## 1. Replace the NumPy import

```python
import cunumpy as xp

a = xp.linspace(0.0, 1.0, 5)
b = xp.sin(a) ** 2 + xp.cos(a) ** 2
print(b, xp.get_backend())  # [1. 1. 1. 1. 1.] numpy
```

`xp` behaves like NumPy: array creation, arithmetic, reductions, `xp.linalg`,
`xp.fft` and `xp.random` are all available. Behind the scenes every attribute
is forwarded to the `array-api-compat` module of the selected library, so the
results are ordinary NumPy (or CuPy) arrays.

## 2. Run the same code on a GPU

Select CuPy before the arrays are created, either from the shell:

```bash
ARRAY_BACKEND=cupy python my_script.py
```

or in the program:

```python
xp.set_backend("cupy")
a = xp.linspace(0.0, 1.0, 5)  # now a cupy.ndarray on the GPU
print(xp.get_backend())       # 'cupy', or 'numpy' if no usable GPU
```

If CuPy cannot be used, CuNumpy falls back to NumPy, so the same script runs on
a laptop and on a GPU node. More in [Choosing a backend](guides/backends.md).

## 3. Write functions that follow their input

The active backend decides where *new* arrays are created. A reusable function
should instead follow the arrays it receives:

```python
def normalize(values):
    array_xp = xp.get_array_module(values)  # numpy- or cupy-compat module
    return values / array_xp.linalg.norm(values)
```

More in [Writing backend-agnostic code](guides/portable-code.md).

## 4. Move data explicitly

Existing arrays never move by themselves. Convert at boundaries such as file
output or plotting:

```python
host = xp.to_numpy(a)      # always a NumPy array on the host
device = xp.to_cupy(host)  # a CuPy array (needs a GPU)
active = xp.to_cunumpy(host)  # whatever backend is active
```

Keep the arrays on the GPU for the whole computation and transfer once. A test
can check that a time step makes no transfer at all:

```python
with xp.profiling.assert_no_transfers():
    step(state, dt)
```

More in [Moving data between host and device](guides/data-movement.md).

## 5. Port a compute kernel

Compiled host kernels (for example with [Pyccel](https://github.com/pyccel/pyccel))
take NumPy arrays. CuNumpy lets you pair each with a CUDA kernel and calls the
one that matches the backend:

```python
AXPY = r"""
extern "C" __global__
void axpy(double a, const double* x, double* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
"""


def axpy(a, x, y, n):  # the host version
    y[:n] += a * x[:n]


kernel = xp.kernels.Kernel(axpy, xp.cuda.CudaKernel(AXPY, "axpy"))

x = xp.arange(1000, dtype=xp.float64)
y = xp.zeros(1000)
kernel(2.0, x, y, 1000, n_threads=1000)  # host on NumPy, CUDA on CuPy
```

More in [Porting kernels to the GPU](kernels/overview.md).

## Which guide do I need?

| I want to ... | Read |
| --- | --- |
| run array code on CPU or GPU with one code base | [Choosing a backend](guides/backends.md), [Backend-agnostic code](guides/portable-code.md) |
| understand when data is copied and avoid slow transfers | [Data movement](guides/data-movement.md) |
| control GPUs, memory pools and streams | [Devices, memory and streams](guides/gpu-devices.md) |
| run one MPI rank per GPU | [Multi-GPU programs with MPI](guides/mpi.md) |
| time GPU code or see it in Nsight | [Timing and profiling](guides/profiling.md) |
| call existing NumPy/Pyccel kernels with GPU arrays | [Host kernels with GPU data](kernels/pyccel-kernel.md) |
| write and launch CUDA kernels from Python | [Writing CUDA kernels](kernels/cuda-kernel.md) |
| port a code base kernel by kernel | [Pairing host and CUDA kernels](kernels/dispatch.md) |
| pass particle or grid data to kernels as one object | [Kernel arguments and structs](kernels/arguments.md) |
| scatter-add into a buffer another library owns | [Accumulation kernels](kernels/accumulation.md) |
| find an illegal memory access | [Debugging CUDA kernels](kernels/debugging.md) |
| test that CPU and GPU kernels agree | [Testing kernels](kernels/testing.md) |
| see everything working together | [Worked examples](examples/index.md) |
