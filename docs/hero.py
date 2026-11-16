"""The figure on the docs home page: vibrating strings computed with cunumpy arrays.

Seven periodic strings carry the wave equation, all at once in one ``(7, n)`` array of the active
backend (NumPy on the CPU, CuPy on a GPU: ``CUNUMPY_BACKEND=cupy``). The leapfrog step runs at a
Courant number of one, where it is exact on the grid, so after ``n`` steps every string is back
where it started and the animation loops without a seam. The snapshots, copied to the host with
``xp.to_numpy``, are drawn with ``plasma_plots.hero`` into ``docs/src/assets/hero.svg``. Rerun
with::

    python docs/hero.py   # or: make docs-hero
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

import cunumpy as xp

strings, n, every = 7, 96, 4
x = xp.linspace(0.0, 8.0, n, endpoint=False)
k = 2 * np.pi / 8.0
row = xp.arange(strings, dtype=xp.float64)[:, None]
# a different mix of modes on each string; the second mode moves the other way
u = 0.12 * xp.sin(k * (row % 3 + 1) * x + row) + 0.07 * xp.cos(
    k * (row % 4 + 2) * x - 2 * row
)
previous = 0.12 * xp.sin(k * (row % 3 + 1) * (x + 8.0 / n) + row) + 0.07 * xp.cos(
    k * (row % 4 + 2) * (x - 8.0 / n) - 2 * row
)

frames = []
for step in range(n + 1):
    if step % every == 0:
        frames.append(xp.to_numpy(u))
    u, previous = xp.roll(u, 1, axis=1) + xp.roll(u, -1, axis=1) - previous, u

backend = xp.get_backend()
fig = HeroFigure(xlim=(0, 8), ylim=(0, 3))
grid = np.r_[xp.to_numpy(x), 8.0][::2]
for s in range(strings):
    level = 0.35 + s * 2.3 / (strings - 1)
    # the last frame is the first again; morph closes the loop itself
    fig.morph(
        grid,
        [level + np.r_[f[s], f[s, 0]][::2] for f in frames[:-1]],
        12.0,
        tone="high" if s == 3 else "accent",
        width=2.0 if s == 3 else 1.4,
        opacity=0.95 if s == 3 else 0.55,
        title=f"string {s + 1}, on {backend}",
    )
fig.save(Path(__file__).parent / "src" / "assets" / "hero.svg")
