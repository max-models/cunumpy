---
title: "Interop"
description: "Hand simulation results to JAX, PyTorch or MLX and back: whole dicts, lists and tuples of arrays, and scalar objectives."
---

The adapters in `cunumpy.interop` (`jax`, `torch`, `mlx`) convert arrays
explicitly between NumPy or CuPy and a machine-learning framework. They share
memory where they can, never change a dtype that was not asked for, and give no
gradients through cunumpy code. Each needs its extra, e.g. `cunumpy[jax]`.

## Nested containers

`tree_to_framework` and `tree_from_framework` convert every array in nested
dicts, lists and tuples (named tuples included) and keep the structure; numbers,
strings and `None` pass through unchanged. The keyword arguments (`device`,
`dtype`, `copy`, `backend`) apply to each array. The JAX adapter maps over any
JAX pytree.

```python
import numpy as np
from cunumpy.interop import jax as cj

cj.enable_x64()
state = {"density": np.ones(64), "fields": (np.zeros(64), np.zeros(64)), "step": 10}
params = cj.tree_to_jax(state)  # jax arrays, same structure, "step" stays 10
back = cj.tree_from_jax(params)  # NumPy arrays again
```

The aliases `tree_to_jax`, `tree_to_torch`, `tree_to_mlx` (and `tree_from_*`)
name the same functions.

## Scalar objectives

`scalar` returns an objective or a diagnostic as a 0-d framework array, which
is what a loss function or `jax.pure_callback` expects. It takes a Python number,
a NumPy or CuPy scalar, or any array with one element, and raises for more.
A Python float counts as float64, so the dtype rules hold: without
`jax_enable_x64` ask for the cast.

```python
loss = cj.scalar(np.sum(back["density"] ** 2))  # 0-d float64 jax array
loss32 = cj.scalar(0.5, dtype="float32")
```
