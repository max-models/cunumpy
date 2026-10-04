# A portable diffusion solver

This script solves the 2D heat equation with an explicit finite-difference
scheme. It uses only array operations, so it needs no kernels: the same file
runs on NumPy and CuPy. It shows the habits that keep such a script fast on the
GPU:

* the backend is chosen once, at start-up, and reported;
* initial data is created on the host with a fixed seed, then moved to the
  device once;
* the time loop has no transfers and no host synchronization;
* diagnostics and output convert explicitly, every `--every` steps only;
* timing waits for the device.

```python
"""heat.py: explicit 2D diffusion, on CPU or GPU."""

import argparse

import numpy as np

import cunumpy as xp


def laplacian(u, dx):
    """5-point Laplacian with periodic boundaries, on u's own backend."""
    array_xp = xp.get_array_module(u)
    return (
        array_xp.roll(u, 1, axis=0)
        + array_xp.roll(u, -1, axis=0)
        + array_xp.roll(u, 1, axis=1)
        + array_xp.roll(u, -1, axis=1)
        - 4.0 * u
    ) / dx**2


def initial_condition(n, seed):
    """Built on the host, so CPU and GPU runs start from identical data."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    gauss = np.exp(-((x[:, None] - 0.5) ** 2 + (x[None, :] - 0.5) ** 2) / 0.01)
    return gauss + 0.01 * rng.standard_normal((n, n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--every", type=int, default=500)
    args = parser.parse_args()

    xp.set_backend("cupy" if args.gpu else "numpy")
    print(
        f"backend={xp.get_backend()} devices={xp.cuda.device_count()} cunumpy={xp.__version__}"
    )

    dx = 1.0 / args.n
    dt = 0.2 * dx**2  # stable for the explicit scheme
    u = xp.to_cunumpy(initial_condition(args.n, seed=0))  # one host-to-device copy

    with xp.profiling.timed_region("time loop") as timing:
        for step in range(1, args.steps + 1):
            u = u + dt * laplacian(u, dx)
            if step % args.every == 0:
                total = float(xp.sum(u)) * dx**2  # one scalar to the host
                print(f"step {step:5d}: integral = {total:.6f}")

    print(f"{timing.elapsed:.3f} s ({timing.elapsed / args.steps * 1e6:.1f} us/step)")
    np.save("heat_final.npy", xp.to_numpy(u))  # one device-to-host copy


if __name__ == "__main__":
    main()
```

Run it:

```bash
python heat.py             # NumPy
python heat.py --gpu       # CuPy, if available; falls back to NumPy otherwise
```

The integral printed every 500 steps is conserved by the periodic scheme, so it
is also a quick check that both backends compute the same thing.

## Variations

* **Check for transfers in a test.** Wrap a few steps in
  `xp.profiling.assert_no_transfers()` to make sure nobody adds a `to_numpy()` to the loop
  later.
* **Profile.** Mark the update with `xp.profiling.nvtx_range("update")` and run
  `nsys profile -t cuda,nvtx python heat.py --gpu`.
* **Use it as a library.** `laplacian()` follows its input via
  `get_array_module()`, so other code can call it with NumPy or CuPy arrays no
  matter which backend is active.
