# Array ordering and strides

CuNumpy supports C-ordered (row-major) and F-ordered (column-major) arrays on
both NumPy and CuPy. Choose the layout when allocating an array, and keep it
consistent with the compiled kernels that consume it.

For a two-dimensional array `a[row, column]`, C order stores each row together;
F order stores each column together. Indexing and shape stay the same. F order
can help code that processes columns of a particle buffer, but measure the
whole workload, including kernels and copies, before choosing a layout.

## Allocate and inspect

```python
import cunumpy as xp

markers = xp.zeros((100, 6), dtype=xp.float64, order="F")
assert markers.flags.f_contiguous

positions = markers[:, 1:4]
assert positions.flags.f_contiguous
assert positions.shape == (100, 3)
```

A block of complete columns of an F-contiguous array remains F-contiguous.
A block of complete rows of a C-contiguous array remains C-contiguous.
Stepped slices such as `markers[::2, :]` generally have neither layout.
Check `a.flags.c_contiguous`, `a.flags.f_contiguous`, and `a.strides` when
debugging a kernel boundary. Strides are measured in bytes. Arrays with
singleton dimensions or no elements can satisfy both contiguity flags.

Use `xp.asfortranarray(a)` to require F order or `xp.ascontiguousarray(a)` to
require C order. These may allocate a copy; a transpose changes the shape and
indexing meaning and is not a substitute for an order conversion.

## Host and device conversions

| Operation | Layout behavior |
| --- | --- |
| `xp.to_numpy(cupy_array)` | F-contiguous input becomes F-contiguous host storage; other input becomes C-contiguous host storage. |
| `xp.to_numpy(numpy_array)` | Uses `numpy.asarray`; an ordinary NumPy array keeps its layout and strides without copying. |
| `xp.to_cupy(numpy_array)` | Uses `cupy.asarray`, preserving C or F order for contiguous input. |
| `xp.to_cunumpy(a)` | Uses the conversion for the active backend. |
| `xp.host_call`, `xp.evaluate_on_host` | Device arguments follow `to_numpy`; returned NumPy arrays follow `to_cupy`. |
| `xp.kernels.PyccelKernel` | Device arguments use the same host conversion as `to_numpy`; declared outputs are copied back into the original device arrays. |

The device-to-host conversion uses `array.get(order="A")`. This preserves
F-contiguous layout, including complete column blocks, but does not preserve
arbitrary strides or shared storage between distinct views. Calling CuPy's
`array.get()` directly defaults to C order; see the
[CuPy array documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html).

The host function itself controls the layout of any new arrays it creates.
Returning a C-ordered result from `host_call` does not make that result F-ordered
just because an input was F-ordered.

```{note}
Preserving F order in CuPy-to-host conversions is an unreleased fix. Earlier
releases used `array.get()` and produced C-ordered host copies. Applications
depending on F order at a host-kernel boundary need a release containing this fix.
```

## Copies, masks, and assignment

`a.copy()` defaults to C order on both NumPy and CuPy. Use `a.copy(order="K")`
to retain the layout of a C- or F-contiguous array, or `a.copy(order="F")` to
require F order. `order="K"` is not a promise to preserve arbitrary strides.

Boolean row masks and fancy indexing create new arrays. Do not rely on those
results retaining F order. For example, a boolean row selection from an
F-ordered marker buffer produces a C-ordered result on CuPy. Normalize an
independent result before passing it to a kernel requiring F order:

```python
alive = markers[:, 0] >= 0
selected = xp.asfortranarray(markers[alive])
assert selected.flags.f_contiguous
```

Assignment into an existing buffer keeps the destination's layout:

```python
markers[...] = markers.copy()  # C-ordered temporary, F-ordered destination
assert markers.flags.f_contiguous
```

Likewise, `xp.algorithms.compact_by_mask(alive, markers)` gathers selected rows
and writes them into the front of the original buffer, preserving that
buffer's layout. Only the returned count of leading rows is valid. However,
`markers[:n_kept]` generally loses F contiguity when it trims rows: the columns
still have the original buffer's stride. Pass the full buffer plus a valid-row
count to a kernel designed for that interface, or make an F-contiguous copy
of the prefix for a kernel that requires one. A stride-aware CUDA array view
can consume the prefix directly.

Check or normalize arrays created by concatenation, file readers, and other
libraries before passing them across a layout-sensitive boundary.

## Compiled kernel boundaries

Pyccel array signatures specify an expected order. A two-dimensional
`"float[:, :]"` argument expects C order; use `"float[:, :](order=F)"` for an
F-ordered argument. Every caller must supply the declared layout, including
temporary arrays and array attributes in argument objects.
`PyccelKernel` preserves F-contiguous device arguments when moving them to
the host; it does not inspect signatures or fix mismatches for you. The same
requirement applies to a `Kernel` using its host fallback without a CUDA version.
See [Host kernels with GPU data](../kernels/pyccel-kernel.md).

CUDA `Array2D<T>` and other `ArrayND<T>` view arguments carry shape and strides,
so indexing through the view supports C order, F order, and strided slices.
The same applies to array view fields in `CudaStruct` and to emulated launches.
Raw pointer arguments and `CArrayND<T>` views require C-contiguous storage.
Hand-written indexing such as `data[row * n_columns + column]` assumes C order;
use the array view's indexing to handle both layouts. See
[Index macros and array views](../kernels/cuda-kernel.md).

Some convenience helpers deliberately require C order:

| Helper | Contract |
| --- | --- |
| `xp.as_device_array` | Returns a C-contiguous device array, copying if needed. |
| `xp.kernels.as_kernel_array` | Returns a C-contiguous array on the side of `like`. |
| `xp.kernels.kernel_output` | Yields a C-contiguous working buffer and copies updates back into the original output when needed. |

For an F-order kernel, allocate or normalize its arguments explicitly instead
of using these helpers to prepare F-ordered inputs.

## Check values and layout separately

Numerical equality does not check memory order. Tests for layout-sensitive
code should assert the host argument's contiguity inside the called function,
as well as checking values. Use a two-dimensional shape with both dimensions
greater than one so C and F order are distinguishable.

The fake CuPy supports these conversion checks without a GPU. Its
`kernel_testing.host_buffer(a)` exposes the existing NumPy storage without a
copy, preserving layout and strides. It does not exercise the `to_numpy`
transfer path; test that path separately. See [Testing kernels](../kernels/testing.md).
