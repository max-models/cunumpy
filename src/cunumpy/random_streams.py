"""Reproducible random numbers for MPI programs on either backend.

A simulation that draws random numbers in many places (initial loading,
injection, collisions) is reproducible only if every draw comes from a seeded
generator, and an MPI run needs a different stream on every rank.
:data:`random_streams` is one generator per process and backend for that::

    import cunumpy as xp

    xp.rng.random_streams.seed(42, rank=comm.Get_rank())  # once, at start-up
    v = xp.rng.random_streams.normal(0.0, v_th, (n, 3))    # anywhere afterwards
    rng = xp.rng.random_streams.generator()               # the Generator itself

Each rank draws the stream ``(seed, rank)`` (a NumPy ``SeedSequence`` with the
rank as spawn key), so a run with the same seed and the same number of ranks
reproduces its results exactly, and the ranks' streams are independent. The
generator of each backend is created on first use from the same stream: a
``numpy.random.Generator`` with the chosen bit generator, or a
``cupy.random.Generator`` (CuPy's default bit generator).

Components with a seed of their own (e.g. a source configured with a ``seed``)
get a separate generator with :meth:`RandomStreams.make_generator`; without one
they share the process generator. Without :meth:`~RandomStreams.seed`, or with
``seed(None)``, the stream is seeded from the operating system.

The draw functions work with NumPy and CuPy generators alike; CuPy's
``Generator`` lacks some of NumPy's methods, and the missing ones are derived
from ``random`` and ``standard_normal``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .xp import get_backend

__all__ = ["BIT_GENERATORS", "RandomStreams", "random_streams"]

#: NumPy bit generators :meth:`RandomStreams.seed` accepts by name.
BIT_GENERATORS = ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64")

_BACKEND_KEYS = {"numpy": 0, "cupy": 1}


class RandomStreams:
    """One seeded random generator per process and backend; see :mod:`cunumpy.random_streams`."""

    def __init__(self) -> None:
        self._sequence: np.random.SeedSequence | None = None
        self._bit_generator = "PCG64"
        self._generators: dict[str, Any] = {}

    def __repr__(self) -> str:
        if self._sequence is None:
            return "RandomStreams(not seeded)"
        return (
            f"RandomStreams(entropy={self._sequence.entropy}, "
            f"spawn_key={self._sequence.spawn_key}, bit_generator={self._bit_generator!r})"
        )

    def seed(
        self, value: int | None, rank: int = 0, bit_generator: str | None = None
    ) -> None:
        """Seed all draws of this process with the stream ``(value, rank)``.

        Replaces the generators of all backends. NumPy's global random state
        (and, on the CuPy backend, CuPy's) is seeded from the same stream, for
        code that still calls ``np.random.*`` or ``xp.random.*`` directly.

        Parameters
        ----------
        value : int | None
            The seed; None seeds from the operating system.
        rank : int
            MPI rank of this process, so that ranks draw independent streams.
        bit_generator : str | None
            The NumPy bit generator, one of :data:`BIT_GENERATORS`; ``PCG64``
            (NumPy's default) if None. CuPy always uses its own default.

        Raises
        ------
        ValueError
            For an unknown `bit_generator`.
        """
        if bit_generator is not None and bit_generator not in BIT_GENERATORS:
            raise ValueError(
                f"Unknown bit generator {bit_generator!r}; use one of "
                f"{', '.join(BIT_GENERATORS)}"
            )
        self._bit_generator = bit_generator or "PCG64"
        self._sequence = np.random.SeedSequence(
            entropy=None if value is None else int(value), spawn_key=(int(rank),)
        )
        self._generators.clear()
        legacy = int(self._sequence.generate_state(1)[0])
        np.random.seed(legacy)
        if get_backend() == "cupy":
            import cupy

            cupy.random.seed(legacy)

    def _backend_sequence(self, backend: str) -> np.random.SeedSequence:
        if self._sequence is None:
            self._sequence = np.random.SeedSequence()
        return np.random.SeedSequence(
            entropy=self._sequence.entropy,
            spawn_key=(*self._sequence.spawn_key, _BACKEND_KEYS[backend]),
        )

    def generator(self, backend: str | None = None) -> Any:
        """The process generator of `backend` (the active backend by default)."""
        backend = get_backend() if backend is None else backend
        if backend not in _BACKEND_KEYS:
            raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}")
        if backend not in self._generators:
            sequence = self._backend_sequence(backend)
            if backend == "numpy":
                bits = getattr(np.random, self._bit_generator)(sequence)
                self._generators[backend] = np.random.Generator(bits)
            else:
                import cupy

                state = int(sequence.generate_state(1, dtype=np.uint64)[0])
                self._generators[backend] = cupy.random.default_rng(state)
        return self._generators[backend]

    def make_generator(
        self, seed: int | None = None, backend: str | None = None
    ) -> Any:
        """A generator for a component: its own if it has a seed, else the process one.

        Parameters
        ----------
        seed : int | None
            The component's own seed; None shares :meth:`generator`.
        backend : str | None
            ``"numpy"`` for a component that draws on the host whatever the
            active backend; the active backend by default.
        """
        if seed is None:
            return self.generator(backend)
        backend = get_backend() if backend is None else backend
        if backend == "numpy":
            return np.random.default_rng(int(seed))
        import cupy

        return cupy.random.default_rng(int(seed))

    def _rng(self, rng: Any) -> Any:
        return self.generator() if rng is None else rng

    def random(self, size: int | tuple[int, ...] | None = None, rng: Any = None) -> Any:
        """Uniform samples in [0, 1) from `rng` (the process generator by default)."""
        return self._rng(rng).random(size=size)

    def standard_normal(
        self, size: int | tuple[int, ...] | None = None, rng: Any = None
    ) -> Any:
        """Standard normal samples from `rng` (the process generator by default)."""
        return self._rng(rng).standard_normal(size=size)

    def normal(
        self,
        loc: Any = 0.0,
        scale: Any = 1.0,
        size: int | tuple[int, ...] | None = None,
        rng: Any = None,
    ) -> Any:
        """Normal samples from `rng` (the process generator by default)."""
        rng = self._rng(rng)
        if hasattr(rng, "normal"):
            return rng.normal(loc=loc, scale=scale, size=size)
        return loc + scale * rng.standard_normal(size=size)

    def uniform(
        self,
        low: Any = 0.0,
        high: Any = 1.0,
        size: int | tuple[int, ...] | None = None,
        rng: Any = None,
    ) -> Any:
        """Uniform samples in [low, high) from `rng` (the process generator by default)."""
        rng = self._rng(rng)
        if hasattr(rng, "uniform"):
            return rng.uniform(low=low, high=high, size=size)
        return low + (high - low) * rng.random(size=size)


#: The random streams of this process.
random_streams = RandomStreams()
