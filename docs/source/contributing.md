# Writing docstrings

Docstrings use the [NumPy format](https://numpydoc.readthedocs.io/en/latest/format.html)
and are the source of the [API reference](reference/index.md). Keep them short:
help text, not a guide. Longer explanations go in a guide page that the
docstring links to.

```python
def cell_offsets(sorted_cells, n_cells):
    """Return the offsets of each cell into sorted cell IDs.

    Cell ``k`` occupies ``[offsets[k], offsets[k + 1])``. The result stays on
    the backend of `sorted_cells`; validation may synchronize on CUDA.

    Parameters
    ----------
    sorted_cells : array of int
        Cell IDs in ``[0, n_cells)``, sorted in nondecreasing order.
    n_cells : int
        Number of cells.

    Returns
    -------
    array of int64
        ``n_cells + 1`` offsets.

    Raises
    ------
    ValueError
        If `sorted_cells` is not sorted or has an ID outside ``[0, n_cells)``.

    See Also
    --------
    segment_boundaries : The runs of sorted keys that are present.

    Examples
    --------
    >>> xp.algorithms.cell_offsets(xp.asarray([0, 0, 2]), 3)
    array([0, 2, 2, 3])
    """
```

## Rules

- **Summary**: one line, imperative mood ("Return", "Count", "Select"),
  ending with a period. Properties and attributes use a noun phrase ("The
  device of the array.").
- **Extended summary**: at most about three sentences, only when needed. Keep
  what is specific to cunumpy: which backend the result lives on, whether the
  call copies, and whether it synchronizes the GPU.
- **Sections**, in this order, empty ones left out: Parameters, Returns (or
  Yields), Raises, See Also, Notes, Examples.
- **Types** are short and conceptual (`array of int`, `str`, `int, optional`,
  `CudaKernel`), not a copy of the annotation. Write `, optional` for
  parameters that have a default and say the default in the description when
  it is not obvious.
- **Raises**: only errors a caller is expected to handle (bad arguments,
  missing CUDA), not every internal check.
- **Classes**: document the constructor parameters in the class docstring;
  `__init__` has no docstring. Public attributes go in an Attributes section.
- **Properties and trivial methods** (`start`, `stop`, `__getitem__`): one
  line, no sections.
- **Private helpers** (names starting with `_`): at most a one-line summary.
- **Cross-references** use the public path: `` :class:`~cunumpy.kernels.CudaKernel` ``,
  not `cunumpy._cuda_kernel.CudaKernel`. Parameter names in single
  backticks, literal code in double backticks.

## Examples

- Every public function and class has an Examples section; methods have one
  when the use is not obvious.
- One example, two to five lines. The examples assume `import cunumpy as xp`
  and `import numpy as np` and run on the NumPy backend (see the root
  `conftest.py`), so don't import those again.
- Examples are tested: `pytest --doctest-modules src/cunumpy`. Lines that need
  a GPU, MPI, Pyccel or Metal get `# doctest: +SKIP`, as do lines that use
  undefined names for illustration. Prefer real, runnable examples.
- Don't paste long CUDA sources; a kernel of a few lines is fine, otherwise
  link the [kernel guides](kernels/overview.md).

## Checking

```bash
ruff check src                            # D rules: summary line, imperative mood, ...
python -m numpydoc lint src/cunumpy/*.py  # sections, documented parameters
python -m pytest --doctest-modules src/cunumpy
cd docs && make html
```
