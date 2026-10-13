"""Reproducible random numbers for MPI programs on either backend.

:data:`random_streams` is one seeded generator per process and backend. Each
rank draws the stream ``(seed, rank)`` (a NumPy ``SeedSequence`` with the rank
as spawn key), so the same seed and number of ranks reproduce a run exactly
and the ranks' streams are independent. The generator of each backend is
created on first use from that stream. See :doc:`/guides/mpi` for the use in
MPI programs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cunumpy.xp import get_backend

__all__ = ["BIT_GENERATORS", "RandomStreams", "random_streams"]

#: NumPy bit generators :meth:`RandomStreams.seed` accepts by name.
BIT_GENERATORS = ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64")

_BACKEND_KEYS = {"numpy": 0, "cupy": 1}


class RandomStreams:
    """One seeded random generator per process and backend.

    Without :meth:`seed`, or with ``seed(None)``, the stream is seeded from the
    operating system. The draw methods work with NumPy and CuPy generators
    alike; :data:`random_streams` is the process-wide instance, and a new
    instance is independent of it.

    See Also
    --------
    get_rng : A fresh, unshared generator for the active backend.

    Examples
    --------
    >>> streams = xp.rng.RandomStreams()
    >>> streams.seed(42, rank=1)
    >>> streams
    RandomStreams(entropy=42, spawn_key=(1,), bit_generator='PCG64')
    >>> streams.uniform(0.0, 1.0, 3)
    array([0.57002868, 0.9724905 , 0.50377011])
    """

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
        self,
        value: int | None,
        rank: int = 0,
        bit_generator: str | None = None,
    ) -> None:
        """Seed all draws of this process with the stream ``(value, rank)``.

        Replaces the generators of all backends. NumPy's global random state
        (and CuPy's on the CuPy backend) is seeded from the same stream, for
        code that calls ``np.random.*`` directly.

        Parameters
        ----------
        value : int or None
            The seed; None seeds from the operating system.
        rank : int, optional
            MPI rank of this process, so that ranks draw independent streams.
        bit_generator : str, optional
            The NumPy bit generator, one of :data:`BIT_GENERATORS`; ``PCG64``
            by default. CuPy always uses its own default.

        Raises
        ------
        ValueError
            For an unknown `bit_generator`.

        Examples
        --------
        >>> xp.rng.random_streams.seed(42, rank=comm.Get_rank())  # doctest: +SKIP
        """
        if bit_generator is not None and bit_generator not in BIT_GENERATORS:
            raise ValueError(
                f"Unknown bit generator {bit_generator!r}; use one of "
                f"{', '.join(BIT_GENERATORS)}",
            )
        self._bit_generator = bit_generator or "PCG64"
        self._sequence = np.random.SeedSequence(
            entropy=None if value is None else int(value),
            spawn_key=(int(rank),),
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
        """Return the process generator of a backend, created on first use.

        Parameters
        ----------
        backend : {"numpy", "cupy"}, optional
            The backend; the active one by default.

        Returns
        -------
        numpy.random.Generator or cupy.random.Generator
            The shared generator of this process and backend.

        Raises
        ------
        ValueError
            For a `backend` other than ``"numpy"`` or ``"cupy"``.
        """
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
        self,
        seed: int | None = None,
        backend: str | None = None,
    ) -> Any:
        """Return a generator for a component: its own if it has a seed, else the shared one.

        Parameters
        ----------
        seed : int, optional
            The component's own seed; None returns :meth:`generator`.
        backend : {"numpy", "cupy"}, optional
            ``"numpy"`` for a component that draws on the host whatever the
            active backend; the active backend by default.

        Returns
        -------
        numpy.random.Generator or cupy.random.Generator
            A new generator seeded with `seed`, or the process generator.
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
        """Draw uniform samples in [0, 1).

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape; None gives a scalar.
        rng : Generator, optional
            The generator to draw from; the process generator by default.

        Returns
        -------
        array or float
            On the backend of the generator.
        """
        return self._rng(rng).random(size=size)

    def standard_normal(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: Any = None,
    ) -> Any:
        """Draw standard normal samples.

        Parameters
        ----------
        size : int or tuple of int, optional
            Output shape; None gives a scalar.
        rng : Generator, optional
            The generator to draw from; the process generator by default.

        Returns
        -------
        array or float
            On the backend of the generator.
        """
        return self._rng(rng).standard_normal(size=size)

    def normal(
        self,
        loc: Any = 0.0,
        scale: Any = 1.0,
        size: int | tuple[int, ...] | None = None,
        rng: Any = None,
    ) -> Any:
        """Draw normal samples, also from CuPy generators without ``normal``.

        Parameters
        ----------
        loc : float or array, optional
            Mean.
        scale : float or array, optional
            Standard deviation.
        size : int or tuple of int, optional
            Output shape; None gives a scalar.
        rng : Generator, optional
            The generator to draw from; the process generator by default.

        Returns
        -------
        array or float
            On the backend of the generator.
        """
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
        """Draw uniform samples in [low, high), also from CuPy generators without ``uniform``.

        Parameters
        ----------
        low, high : float or array, optional
            The bounds, 0 and 1 by default.
        size : int or tuple of int, optional
            Output shape; None gives a scalar.
        rng : Generator, optional
            The generator to draw from; the process generator by default.

        Returns
        -------
        array or float
            On the backend of the generator.
        """
        rng = self._rng(rng)
        if hasattr(rng, "uniform"):
            return rng.uniform(low=low, high=high, size=size)
        return low + (high - low) * rng.random(size=size)


#: The random streams of this process.
random_streams = RandomStreams()


def get_rng(seed: int | None = None) -> Any:
    """Return a new random generator for the active backend.

    ``default_rng(seed)`` of NumPy or CuPy. The APIs are similar, but the same
    seed does not give the same numbers on both backends.

    Parameters
    ----------
    seed : int, optional
        The seed; None seeds from the operating system.

    Returns
    -------
    numpy.random.Generator or cupy.random.Generator
        A generator independent of :data:`random_streams`.

    Examples
    --------
    >>> xp.rng.get_rng(seed=7).random(2)
    array([0.62509547, 0.8972138 ])
    """
    if get_backend() == "cupy":
        import cupy as cp

        return cp.random.default_rng(seed)
    return np.random.default_rng(seed)
