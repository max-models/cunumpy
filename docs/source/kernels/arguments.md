# Kernel arguments and structs

Kernels of real simulations take many arguments: the marker array, its shape,
the grid spacing, spline degrees, knot vectors, a dozen parameters. Listing
them at every call site is error-prone, and every new field means editing every
kernel signature. CuNumpy groups them for CUDA kernels:

| Tool | CUDA kernel sees | Use when |
| --- | --- | --- |
| `CudaArguments` | several parameters, flattened | grouping a few device arrays and scalars |
| `CudaStruct` | one C struct parameter | many fields; one definition shared by all CUDA kernels |
| `CudaStructArguments` | one C struct parameter | the struct as a class: an object with attributes, built once and reused |

CuNumpy never converts an argument object between a host form and a CUDA
form. A host kernel gets its own argument object (e.g. a pyccel class of NumPy
arrays), a CUDA kernel gets a CUDA one, and the code that owns the arrays
decides which one to build; see [Host and CUDA argument classes](#host-and-cuda-argument-classes).

## `CudaArguments`: flatten into several parameters

Any object with a `__cuda_args__()` method returning a tuple is replaced by
that tuple when passed to a `CudaKernel`. Subclassing `CudaArguments` gives you
the method for free:

```python
import numpy as np

import cunumpy as xp


class DeviceParticles(xp.arguments.CudaArguments):
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
push = xp.kernels.CudaKernel(PUSH, "push")
particles = DeviceParticles(x, v)
push(0.1, particles, n_threads=particles.positions.shape[0])  # -> push(0.1, x, v, n)
```

The values may include packed structs (`CudaStructValue.packed`) for the
struct parameters of the kernel.

Build the object once and reuse it. `as_device_array()` references existing
device arrays of the right dtype and copies everything else exactly once (see
[Data movement](../guides/data-movement.md), section "Build device arguments once").

## `CudaStruct`: one C struct, defined once

A `CudaStruct` defines a C struct in Python: its C declaration for the kernel
source, its exact memory layout, and a packer for values. The kernel takes the
struct by value as one parameter:

```python
Particles = xp.arguments.CudaStruct(
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
push = xp.kernels.CudaKernel(PUSH, "push", structs=[Particles])

value = Particles(x=x, v=v, n=x.size, charge=-1.0)
push(value, 0.1, n_threads=x.size)
```

Fields may be scalars, pointers to scalar types (or `void*`), and array views
`Array1D<T>` to `Array16D<T>`. Packing checks every field like a kernel
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
class CudaMarkerArguments(xp.arguments.CudaStructArguments):
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


xp.arguments.write_cuda_header("kernels/marker_args.cuh", [CudaMarkerArguments.struct])
push = xp.kernels.CudaKernel.from_file(
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
  class CudaMarkerArguments(xp.arguments.CudaStructArguments):
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
* For the host kernels, write the matching host class and let the owner choose,
  see [Host and CUDA argument classes](#host-and-cuda-argument-classes).

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


MarkerArgs = xp.arguments.CudaStruct.from_signature(
    MarkerArguments.__init__, "MarkerArgs"
)
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
C types, `"float[:, :]"` to `Array2D<double>` (1 to 16 dimensions). `Final[...]`
and `const` are ignored. `scalar_names={"float": "float"}` switches to single
precision. Parameters without a mappable annotation raise `ValueError`.

If the module of the argument class is compiled by pyccel, importing it gives
the compiled class, whose `__init__` has no Python signature.
`CudaStruct.from_pyccel_class(path, class_name, name)` parses the `.py` source
with `ast` instead (never importing it), names each field after the attribute
the parameter is stored in (`self.first_init_idx = first_pusher_idx` gives a
field `first_init_idx`), and skips the parameters in `exclude=`:

```python
MarkerArgs = xp.arguments.CudaStruct.from_pyccel_class(
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
`CArray16D<T>` instead. The view holds a pointer and the shape, no strides, and
`m.markers(ip, 0)` is `data[ip * shape[1] + 0]`: the last index is always the
fast one, as in the row-major memory the host code uses.

```python
MarkerArgs = xp.arguments.CudaStruct.from_pyccel_class(
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
xp.arguments.write_cuda_header("kernels/marker_args.cuh", [MarkerArgs, DomainArgs])
```

The header gets an include guard (`MARKER_ARGS_CUH`), the `array_view.cuh`
include when needed, and the definitions. `MarkerArgs.to_header(path)` does the
same for one struct. A test keeps the committed file in sync:

```python
from pathlib import Path


def test_marker_args_header_is_up_to_date(tmp_path):
    generated = xp.arguments.write_cuda_header(
        tmp_path / "marker_args.cuh", [MarkerArgs, DomainArgs]
    )
    assert Path("kernels/marker_args.cuh").read_text() == generated
```

## Host and CUDA argument classes

A host kernel compiled with Pyccel often receives a group of arrays as an
instance of an argument class. Write its CUDA counterpart as a
`CudaStructArguments` subclass with the same constructor and attributes, and
let the owner of the arrays create the one that matches the backend:

```python
from my_sim.kernel_arguments.pusher_args_kernels import MarkerArguments  # pyccel


class CudaMarkerArguments(xp.arguments.CudaStructArguments):
    """CUDA version of MarkerArguments: same constructor, same attributes."""

    struct_name = "MarkerArgs"
    fields = (("markers", "Array2D<double>"), ("n_markers", "int"))

    def __init__(self, markers, n_markers):
        self.markers = markers  # a CuPy array
        self.n_markers = n_markers
        self.pack()


class Particles:
    def __init__(self, markers):
        self.markers = markers
        args_class = CudaMarkerArguments if xp.is_gpu(markers) else MarkerArguments
        self.args_markers = args_class(markers, markers.shape[0])


push = xp.kernels.Kernel.from_folder("my_sim.kernels.push")
push(particles.args_markers, dt)  # the pyccel object on NumPy, the struct on CuPy
```

* `Kernel` and `PyccelKernel` pass the host object to the host kernel as it
  is; `CudaKernel` flattens a CUDA argument object with `__cuda_args__()`.
* Pass the CUDA object to the host kernel, or the pyccel object to the CUDA
  kernel, and the kernel's own argument checks raise. Nothing is copied
  behind your back.
* A test can check that the two classes of a pair stay in sync, by comparing
  `CudaStruct.from_pyccel_class(...)` (the pyccel constructor) with
  `CudaMarkerArguments.struct.fields`.

## Choosing

* Start with plain arguments. Group when the same set of five or more values
  appears in several kernels.
* Use a `CudaStruct` when many CUDA kernels take the same group, or when
  signatures become too long to read and keep in sync.
* When the host kernels take an argument class, write a `CudaStructArguments`
  with the same constructor and attributes, and build one of the two per
  backend.
* Generate the struct with `from_signature` or `from_pyccel_class` when a host
  argument class exists, or test that the two field lists match, so the two
  cannot drift apart.
