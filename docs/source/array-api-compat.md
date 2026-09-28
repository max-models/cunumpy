# Why CuNumpy uses `array-api-compat`

You can use CuNumpy without importing `array-api-compat` yourself. It is a
dependency that sits between CuNumpy and the array libraries it selects. This
page explains what that layer does and when you might notice it.

## Start with the three pieces

**NumPy** stores arrays in regular computer memory and performs operations on
the CPU. **CuPy** offers many similar operations on arrays stored on an NVIDIA
GPU. Their APIs overlap substantially, but the same function name does not
always accept the same arguments or follow the same rules.

The **Python Array API standard** describes a shared set of array operations
and their expected behavior. It is a specification, not an array library that
stores data or runs calculations. [`array-api-compat`](https://data-apis.org/array-api-compat/)
provides small compatibility modules for existing libraries, including NumPy
and CuPy. Those modules expose the underlying libraries' functions while
adjusting operations covered by the standard to behave more consistently.

CuNumpy uses those modules as its two backends:

| CuNumpy backend | Module CuNumpy calls | Where the array lives |
| --- | --- | --- |
| `"numpy"` | `array_api_compat.numpy` | CPU memory |
| `"cupy"` | `array_api_compat.cupy` | GPU memory |

For example, after `xp.set_backend("numpy")`, `xp.asarray(...)` is resolved
through the NumPy compatibility module. After `xp.set_backend("cupy")`, it is
resolved through the CuPy compatibility module. You still write `import
cunumpy as xp` and call `xp.asarray(...)`; CuNumpy chooses the module.

## Why add this layer?

Without a compatibility layer, a program that switches between NumPy and
CuPy must account for differences in shared operations. CuNumpy uses
`array-api-compat` so those operations have a more consistent interface. For
example, the compatibility namespace supports the Array API's `device`
argument on `asarray` for both backends:

```python
import cunumpy as xp

with xp.use_backend("numpy"):
    values = xp.asarray([1, 2, 3], device=None)
```

`array-api-compat` also supplies array-type checks used by CuNumpy to tell
whether a particular value is a CuPy array. That is how
`xp.get_array_backend(values)` and helpers such as `xp.is_gpu(values)` can
inspect an array independently of the currently selected backend.

This layer does **not** create a third kind of array. A NumPy-backed result is
still a NumPy array; a CuPy-backed result is still a CuPy array. It also does
not install CuPy or CUDA, move existing arrays when the global backend
changes, or make every NumPy operation available in CuPy. Functions outside
the shared standard may still differ between libraries.

## When should you think about it?

For code that creates arrays and works entirely on the active backend, you
usually do not need to think about the compatibility layer:

```python
import cunumpy as xp

xp.set_backend("numpy")
values = xp.arange(5)
print(xp.sum(values))
```

It becomes useful when a function receives an array created elsewhere. The
global backend may be NumPy even though the argument is a CuPy array, or the
other way around. `xp.get_array_module(array)` returns the matching
compatibility module for that *array*, so operations inside the function use
the correct library:

```python
def mean_center(values):
    array_xp = xp.get_array_module(values)
    return values - array_xp.mean(values)
```

`array_xp` is `array_api_compat.numpy` for a NumPy array and
`array_api_compat.cupy` for a CuPy array. The returned value stays on the
same backend as `values`. You do not need to import either compatibility
module directly for this pattern.

If a function combines several inputs, first check that they live on the
same backend with `xp.assert_same_backend(a, b)`, or convert them explicitly
with `xp.to_numpy()`, `xp.to_cupy()`, or `xp.to_cunumpy()`. The compatibility
layer does not make mixed CPU/GPU arithmetic automatic.

## What to remember

* Use `xp.get_backend()` to ask which backend CuNumpy currently selects.
* Use `xp.get_array_backend(array)` to ask where a particular array lives.
* Use `xp.get_array_module(array)` when a function should follow its input
  array rather than the global selection.
* Use explicit conversion helpers when data must move between CPU and GPU.

The [`array-api-compat` documentation](https://data-apis.org/array-api-compat/)
explains the compatibility modules and the Array API standard in more depth.
