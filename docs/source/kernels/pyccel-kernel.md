# Host kernels with GPU data

`PyccelKernel` adapts a function that expects NumPy arrays so it can be called
with CuPy arrays. It is named after Pyccel, whose compiled functions accept
only NumPy arrays, but it wraps any callable: a Numba function, a C extension,
or plain Python. It does not compile anything.

## What it does

```python
import cunumpy as xp


def smooth(field, out):
    out[1:-1] = 0.25 * field[:-2] + 0.5 * field[1:-1] + 0.25 * field[2:]
    out[0], out[-1] = field[0], field[-1]


smooth_kernel = xp.kernels.PyccelKernel(smooth, outputs=(1,))
```

On every call it decides whether conversion is needed (the active backend is
CuPy, or a CuPy array is among the arguments):

* **No conversion** (NumPy backend, NumPy arrays): the function is called
  directly. Identity, mutation, return values and exceptions behave as without
  the wrapper; the overhead is a type check per argument.
* **Conversion**: every CuPy array in the arguments is copied to the host, the
  function runs on the host copies, the arrays it may have written are copied
  back into the original CuPy arrays, and returned NumPy arrays are converted to
  CuPy.

```python
with xp.use_backend("cupy"):
    field = xp.random.random(1000)
    out = xp.empty_like(field)
    smooth_kernel(field, out)  # out is updated on the device
```

Device-to-host conversion preserves F-contiguous arrays. The host kernel must
accept that order: the wrapper does not adapt arrays to a compiled signature.
See [Array ordering and strides](../guides/array-ordering.md) for Pyccel signatures
and the operations that can change layout.

## Declare the outputs

The wrapper cannot know which arguments the function writes. By default it
copies back *every* converted array, which is correct but doubles the
transfers. `outputs` lists the arguments that may be written:

```python
xp.kernels.PyccelKernel(smooth, outputs=(1,))  # positional argument 1
xp.kernels.PyccelKernel(update, outputs=(0, -1))  # first and last argument
xp.kernels.PyccelKernel(solve, outputs=("out",))  # solve(a, b, out=out)
xp.kernels.PyccelKernel(norm, outputs=())  # writes nothing
```

* Arguments are declared by index (negative indices count from the end) or by
  name. A name finds the argument also when it is passed positionally, and an
  index also when it is passed as a keyword, as long as the parameter names are
  known: from the Python signature, from `parameters=[...]`, or, for a kernel in a
  `Kernel`, from its host function. For a compiled function without any of those
  the two forms are not interchangeable.
* `xp.kernels.outputs_from_annotations(function)` reads the outputs from the
  annotations: every parameter that is not `Final`, `const` or a scalar. A
  `Kernel.from_folder(..., outputs="annotations")` (and `KernelCatalog.from_package`)
  applies it to the host function of each kernel folder.
* A container or object declared as output has all its arrays copied back.
* **A missing declaration is a silent bug**: if the function writes an
  argument that is not declared, the device array keeps its old values. When in
  doubt, leave `outputs=None`.

## Arrays inside containers and objects

Lists, tuples and dicts are traversed recursively, so `kernel([x, y], {"v": v})`
works. Instances of your own classes are traversed only if their class's module
starts with one of the `object_modules` prefixes:

```python
kernel = xp.kernels.PyccelKernel(push, object_modules=("my_simulation.",), outputs=(0,))
kernel(particles, dt)  # particles.positions etc. are converted
```

Such objects are shallow-copied and their array attributes replaced on the
copy, so the caller's object keeps referencing the device arrays. Objects of
other modules are passed through unchanged.

Other details:

* **Aliasing is preserved.** If the same device array appears twice, the host
  function sees one host array twice.
* **Return values**: NumPy arrays (also inside returned tuples and lists) are
  converted to CuPy; `is_array` customizes which return values count as arrays
  (for NumPy subclasses).
* **`use_cupy`** forces conversion on (`True`) or off (`False`); the default
  `None` decides per call.

## Costs and when to move on

Each converted call copies every input array to the host and every output back.
For a kernel called every time step with large arrays, this dominates the run
time. `count_transfers()` records one `kernel_conversion` event per converted
call, naming the kernel and the number of arrays:

```python
with xp.profiling.count_transfers() as counter:
    step(state, dt)
for event in counter.kernel_conversion_calls:
    print(event.where, event.description)
```

`PyccelKernel` is the right tool for:

* getting a code base to run on the GPU backend before any kernel is ported;
* kernels outside the time loop (setup, I/O, rare diagnostics);
* the fallback of a `Kernel` that has no CUDA version yet.

For kernels inside the time loop, write a CUDA version ([Writing CUDA
kernels](cuda-kernel.md)) and pair it with the host kernel ([Pairing host and
CUDA kernels](dispatch.md)).

## Pyodide

In Pyodide only the no-conversion path exists. Plain Python functions can still
be wrapped, so code that uses `PyccelKernel` runs in the browser; see
[Pyodide](../pyodide.md).
