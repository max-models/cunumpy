# Kernel arguments and structs

Kernels of real simulations take many arguments: the marker array, its shape,
the grid spacing, spline degrees, knot vectors, a dozen parameters. Listing
them at every call site is error-prone, and every new field means editing every
kernel signature. CuNumpy offers three ways to pass a group of values as one
argument:

| Tool | Host kernel sees | CUDA kernel sees | Use when |
| --- | --- | --- | --- |
| `CudaArguments` | (not used) | several parameters, flattened | grouping device arguments only |
| `KernelArguments` | one object (`__host_args__()`) | several parameters (`__cuda_args__()`) | the host kernel takes an argument class, the CUDA kernel flat parameters |
| `CudaStruct` | (not used) | one C struct parameter | many fields; one definition shared by all CUDA kernels |
| `CudaStructArguments` | (not used) | one C struct parameter | the struct as a class: an object with attributes, built once and reused |

They combine: a `KernelArguments` object can return a `CudaStruct` value from
`__cuda_args__()`.

## `CudaArguments`: flatten into several parameters

Any object with a `__cuda_args__()` method returning a tuple is replaced by
that tuple when passed to a `CudaKernel`. Subclassing `CudaArguments` gives you
the method for free:

```python
import numpy as np

import cunumpy as xp


class DeviceParticles(xp.cuda.CudaArguments):
    def __init__(self, positions, velocities):
        self.positions = xp.as_device_array(
            positions, np.float64, ndim=2, name="positions"
        )
        self.velocities = xp.as_device_array(
            velocities, np.float64, ndim=2, name="velocities"
        )
        super().__init__(self.positions, self.velocities, self.positions.shape[0])


PUSH = r"""
extern "C" __global__
void push(double dt, double* x, const double* v, int n) { ... }
"""
push = xp.cuda.CudaKernel(PUSH, "push")
particles = DeviceParticles(x, v)
push(0.1, particles, n_threads=particles.positions.shape[0])  # -> push(0.1, x, v, n)
```

Build the object once and reuse it. `as_device_array()` references existing
device arrays of the right dtype and copies everything else exactly once (see
[Data movement](../guides/data-movement.md), section "Build device arguments once").

## `KernelArguments`: one object, two forms

A host kernel compiled with Pyccel often receives a group of arrays as an
instance of an argument class, while the CUDA kernel receives the arrays as
separate parameters. `KernelArguments` lets one object stand for both, so the
call site never branches on the backend:

```python
class MarkerArguments:  # the host argument class, e.g. compiled with Pyccel
    def __init__(self, markers: "float[:, :]", n_markers: int):
        self.markers = markers
        self.n_markers = n_markers


class ParticleArguments(xp.kernels.KernelArguments):
    def __init__(self, particles):
        self._particles = particles
        self._host = None
        self._cuda = None

    def __host_args__(self):
        if self._host is None:
            markers = self._particles.markers
            self._host = MarkerArguments(markers, markers.shape[0])
        return self._host

    def __cuda_args__(self):
        if self._cuda is None:
            markers = self._particles.markers
            self._cuda = (markers, markers.shape[0], markers.shape[1])
        return self._cuda


args = ParticleArguments(particles)
push(args, dt, n_threads=n_markers)  # a Kernel: same call on both backends
```

* `Kernel` (on the NumPy backend) and `PyccelKernel` replace the argument with
  `__host_args__()`; `CudaKernel` flattens `__cuda_args__()`.
* Only top-level arguments are resolved, not objects inside lists or dicts.
* Build both forms lazily, as above: a CPU run never builds device arguments, a
  GPU run never builds the host object.
* **Invalidate the cache** when the underlying arrays are replaced (resizing,
  `deepcopy`, unpickling): reset the stored forms or create a new arguments
  object. Stale cached arguments point at the old arrays.
* `xp.kernels.resolve_host_args(args, kwargs)` applies the host replacement, for code
  that calls host kernels without `Kernel`.

## `CudaStruct`: one C struct, defined once

A `CudaStruct` defines a C struct in Python: its C declaration for the kernel
source, its exact memory layout, and a packer for values. The kernel takes the
struct by value as one parameter:

```python
Particles = xp.cuda.CudaStruct(
    "Particles",
    [("x", "double*"), ("v", "double*"), ("n", "long long"), ("charge", "double")],
)

PUSH = (
    Particles.declaration
    + r"""
extern "C" __global__ void push(Particles p, double dt) {
    long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i < p.n) p.x[i] += dt * p.charge * p.v[i];
}
"""
)
push = xp.cuda.CudaKernel(PUSH, "push", structs=[Particles])

value = Particles(x=x, v=v, n=x.size, charge=-1.0)
push(value, 0.1, n_threads=x.size)
```

Fields may be scalars, pointers to scalar types (or `void*`), and array views
`Array1D<T>` to `Array4D<T>`. Packing checks every field like a kernel
argument: pointers need C-contiguous CuPy arrays of the declared dtype, scalars
are range-checked and cast. Adding a field means editing the one Python
definition; kernels that use the struct pick it up.

The packed value (`CudaStructValue`) holds only device *addresses*. It keeps
references to the arrays so they stay alive, but if you replace an array (for
example `particles.x = new_array`), pack a new value. `value["charge"]` reads a
field back; `Particles.dtype` is the NumPy structured dtype of the layout.

`CudaKernel(..., structs=[Particles])` also checks that a struct definition in
the kernel source matches the Python definition, so a hand-edited header that
drifted raises a `ValueError` instead of reading fields at wrong offsets.

### `CudaStructArguments`: the struct as a class

When the device arguments are an object of their own (built once per particle
species, domain or grid, and kept next to the host argument object), subclass
`CudaStructArguments`. The class declares the struct, the instance holds the
field values as attributes and is passed to kernels as it is:

```python
class CudaMarkerArguments(xp.cuda.CudaStructArguments):
    struct_name = "MarkerArgs"
    fields = (
        ("markers", "double*"),
        ("valid_mks", "bool*"),
        ("n_markers", "int"),
        ("n_cols", "int"),
        ("weight_idx", "int"),
    )

    def __init__(self, markers, valid_mks, weight_idx):
        self.markers = xp.as_device_array(markers, np.float64, ndim=2, name="markers")
        self.valid_mks = xp.as_device_array(
            valid_mks, np.bool_, ndim=1, name="valid_mks"
        )
        self.n_markers, self.n_cols = self.markers.shape
        self.weight_idx = weight_idx
        self.pack()


xp.cuda.write_cuda_header("kernels/marker_args.cuh", [CudaMarkerArguments.struct])
push = xp.cuda.CudaKernel.from_file(
    "kernels/push_cuda.cu", structs=[CudaMarkerArguments.struct]
)

args = CudaMarkerArguments(markers, valid_mks, weight_idx=6)
push(args, dt, n_threads=args.n_markers)
```

* The `CudaStruct` is built when the class is defined (`CudaMarkerArguments.struct`),
  so a bad field type raises at import, not at the first launch.
* `pack()` checks every field like `CudaStruct` does.
* The struct is packed again automatically, at the next launch, when a field
  attribute changed: another array (address, shape or strides) or another
  scalar value. Make the fields properties when the arrays belong to another
  object that replaces them, e.g. a particle container that resizes its marker
  array:

  ```python
  class CudaMarkerArguments(xp.cuda.CudaStructArguments):
      struct_name = "MarkerArgs"
      fields = (("markers", "Array2D<double>"), ("n_markers", "int"))

      def __init__(self, particles):
          self._particles = particles
          self.pack()

      @property
      def markers(self):
          return self._particles.markers

      @property
      def n_markers(self):
          return self._particles.markers.shape[0]
  ```

  Without the properties, the object keeps the old array alive and kernels
  keep working on it: assign the new array to the attribute instead.
* Copies and unpickled objects are packed again from their own arrays, so a
  `deepcopy` never points at the device memory of the original.
* When the same call site must also reach a host kernel, pair it with the host
  argument object in a `KernelArguments`: `__host_args__()` returns the host
  object, `__cuda_args__()` returns `cuda_args.__cuda_args__()`.

### `PyccelStructArguments`: a pyccel host class and a struct

When the host kernels take a pyccel-compiled argument class, that class cannot
inherit from `CudaStructArguments` (or anything else). `PyccelStructArguments`
holds it instead: the same object is passed to a `Kernel` on both backends, and
arrives as the pyccel object on the host path and as the struct on the device
path:

```python
from my_sim.kernel_arguments import pusher_args_kernels  # compiled by pyccel


class MarkerArguments(xp.kernels.PyccelStructArguments):
    struct_name = "MarkerArgs"
    fields = (("markers", "Array2D<double>"), ("Np", "long long"), ("n_markers", "int"))
    host_class = pusher_args_kernels.MarkerArguments
    host_fields = ("markers", "Np")  # its constructor arguments, in order

    def __init__(self, markers, Np):
        self.markers = markers  # NumPy or CuPy, whatever the owner has
        self.Np = Np
        self.n_markers = markers.shape[0]
        if self.has_device_arrays():
            self.pack()  # fail early on a bad device array


args = MarkerArguments(particles.markers, Np)
push(args, dt, n_threads=args.n_markers)  # Kernel: same call on both backends
```

* `__host_args__()` builds `host_class(*host_fields)` once and again when one of
  those attributes was replaced (a resized array, a changed scalar). The host
  object is not pickled; a copy or an unpickled object rebuilds it.
* On the CuPy backend the attributes are device arrays, and there is no host
  form: `__host_args__()` raises. Set `host_copies = True` on a class whose host
  kernels only *read* the arrays (an evaluation, not a push): the host object
  is then built from host copies, which `count_transfers()` reports, and what
  the host kernel writes is not copied back.
* Objects holding host arrays are copied and pickled without packing; the
  struct is only built from device arrays.

### Check the layout against the compiler

Values are packed with the NumPy dtype of the struct. If the compiler lays the
struct out differently (a hand-edited header, a `#pragma pack`, another
compiler such as hiprtc), kernels read fields at the wrong offsets without any
error. One GPU test per struct catches it:

```python
def test_marker_args_layout():
    CudaMarkerArguments.struct.verify_layout()  # the Python declaration
    CudaMarkerArguments.struct.verify_layout(  # the committed header
        "marker_args.cuh", include_dirs=["kernels"]
    )
```

### Generate the struct from the host argument class

If the host kernels already use an annotated argument class (Pyccel style), the
struct can be derived from it, making the Python class the single definition of
the arguments on host and device:

```python
class MarkerArguments:
    def __init__(self, markers: "float[:, :]", n_markers: int, valid: "bool[:]"): ...


MarkerArgs = xp.cuda.CudaStruct.from_signature(MarkerArguments.__init__, "MarkerArgs")
print(MarkerArgs.declaration)
```

```c
struct MarkerArgs {
    Array2D<double> markers;
    long long n_markers;
    Array1D<bool> valid;
};
```

Mappings: `float` to `double`, `int` to `long long` (Pyccel integers are
64-bit; change with `int_type=`), `bool` to `bool`, NumPy scalar types to their
C types, `"float[:, :]"` to `Array2D<double>` (1 to 3 dimensions). `Final[...]`
and `const` are ignored. `scalar_names={"float": "float"}` switches to single
precision. Parameters without a mappable annotation raise `ValueError`.

If the module of the argument class is compiled by pyccel, importing it gives
the compiled class, whose `__init__` has no Python signature.
`CudaStruct.from_pyccel_class(path, class_name, name)` parses the `.py` source
with `ast` instead (never importing it), names each field after the attribute
the parameter is stored in (`self.first_init_idx = first_pusher_idx` gives a
field `first_init_idx`), and skips the parameters in `exclude=`:

```python
MarkerArgs = xp.cuda.CudaStruct.from_pyccel_class(
    "my_sim/kernel_arguments/pusher_args_kernels.py", "MarkerArguments", "MarkerArgs"
)
```

Array view fields accept non-contiguous arrays, and the kernel indexes them like
the host kernel does:

```c
#include "marker_args.cuh"
#include <cunumpy/index.cuh>

extern "C" __global__ void push(MarkerArgs m, double dt) {
    CUNUMPY_THREAD_1D(ip, m.n_markers);
    if (m.valid(ip)) m.markers(ip, 0) += dt * m.markers(ip, 3);
}
```

### Contiguous views: `CArray2D<T>`

`Array2D<T>` takes any view, so its strides are only known at run time and a
kernel cannot tell which index is the fast one. When an array is always
C-contiguous (a marker array, a grid), declare it as `CArray1D<T>` to
`CArray4D<T>` instead. The view holds a pointer and the shape, no strides, and
`m.markers(ip, 0)` is `data[ip * shape[1] + 0]`: the last index is always the
fast one, as in the row-major memory the host code uses.

```python
MarkerArgs = xp.cuda.CudaStruct.from_pyccel_class(
    "my_sim/kernel_arguments/pusher_args_kernels.py",
    "MarkerArguments",
    "MarkerArgs",
    contiguous=["markers"],  # or True for every array field
)
```

```c
struct MarkerArgs {
    CArray2D<double> markers;
    long long n_markers;
    Array1D<bool> valid;
};
```

* A non-contiguous array (e.g. `markers[:, 0:3]`) raises `TypeError` at
  packing or launch. It is **not** copied with `ascontiguousarray`, since
  the kernel would write into the copy and the result would be lost.
  Use `Array2D<T>` for arguments that are sometimes views.
* A `CArrayND<T>` converts to an `ArrayND<T>`, so device helper functions
  written for strided views also take it.
* `contiguous=` works the same in `CudaStruct.from_signature`.

### Write the struct to a header

Kernels in `.cu` files include the struct from a header. Generate it from the
Python definition and commit it:

```python
xp.cuda.write_cuda_header("kernels/marker_args.cuh", [MarkerArgs, DomainArgs])
```

The header gets an include guard (`MARKER_ARGS_CUH`), the `array_view.cuh`
include when needed, and the definitions. `MarkerArgs.to_header(path)` does the
same for one struct. A test keeps the committed file in sync:

```python
from pathlib import Path


def test_marker_args_header_is_up_to_date(tmp_path):
    generated = xp.cuda.write_cuda_header(
        tmp_path / "marker_args.cuh", [MarkerArgs, DomainArgs]
    )
    assert Path("kernels/marker_args.cuh").read_text() == generated
```

## Choosing

* Start with plain arguments. Group when the same set of five or more values
  appears in several kernels.
* Use `KernelArguments` when the host kernels already take argument objects;
  it keeps the call sites identical.
* Use a `CudaStruct` when many CUDA kernels take the same group, or when
  signatures become too long to read and keep in sync.
* Generate the struct with `from_signature` when a host argument class exists,
  so the two cannot drift apart.
