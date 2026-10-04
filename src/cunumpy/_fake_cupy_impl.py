# The body of the fake ``cupy`` module, see cunumpy/_fake_cupy.py. It is executed
# into a fresh module named ``cupy`` (so classes defined here are ``cupy.ndarray``
# etc.), never imported as cunumpy._fake_cupy_impl.

import builtins
import operator
import sys as _sys
import types

import numpy as _np

__version__ = "0.0.0+cunumpy-fake"
__cunumpy_fake__ = True

_HOST = _np.ndarray


def _err(obj, where=""):
    return TypeError(
        f"Unsupported type {type(obj)}{where} (fake CuPy: host arrays/lists are "
        "not accepted)"
    )


class ndarray:
    __slots__ = ("__weakref__", "_a")
    __array_priority__ = 100

    def __init__(self, a):
        assert isinstance(a, _HOST)
        object.__setattr__(self, "_a", a)

    # --- conversions ---------------------------------------------------------
    def __array__(self, *args, **kwargs):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. Please use "
            "`.get()` to construct a NumPy array explicitly."
        )

    def get(self, stream=None, order="C", out=None, blocking=True):
        if out is not None:
            if out.shape != self._a.shape or out.dtype != self._a.dtype:
                raise ValueError("out must match the device array shape and dtype")
            _np.copyto(out, self._a)
            return out
        return self._a.copy(order=order)

    def set(self, arr, *args, **kwargs):
        self._a[...] = arr

    def __reduce__(self):
        return (_from_host, (self._a.copy(),))

    @property
    def __cuda_array_interface__(self):
        a = self._a
        return {
            "shape": a.shape,
            "typestr": a.dtype.str,
            "data": (a.ctypes.data, False),
            "version": 3,
            "strides": None if a.flags.c_contiguous else a.strides,
        }

    @property
    def data(self):
        return types.SimpleNamespace(ptr=self._a.ctypes.data)

    @property
    def device(self):
        return cuda.Device(0)

    # --- attributes ----------------------------------------------------------
    def __getattr__(self, name):
        # no __array_interface__ etc.: NumPy must not see the host buffer
        if name.startswith("__"):
            raise AttributeError(name)
        attr = getattr(self._a, name)
        if callable(attr):
            return _wrap_callable(attr, strict=True, name=f"ndarray.{name}")
        return _wrap(attr)

    def __setattr__(self, name, value):
        setattr(self._a, name, _unwrap(value))

    # --- python protocol -----------------------------------------------------
    def __len__(self):
        return len(self._a)

    def __iter__(self):
        for x in self._a:
            yield _wrap(x)

    def __repr__(self):
        return "fakecupy." + repr(self._a)

    def __str__(self):
        return str(self._a)

    def __format__(self, spec):
        return format(self._a.item() if self._a.ndim == 0 else self._a, spec)

    def __bool__(self):
        return bool(self._a)

    def __int__(self):
        return int(self._a)

    def __float__(self):
        return float(self._a)

    def __complex__(self):
        return complex(self._a)

    def __index__(self):
        return operator.index(self._a.item() if self._a.ndim == 0 else self._a)

    __hash__ = None

    def __contains__(self, x):
        return _unwrap(x) in self._a

    def __getitem__(self, key):
        return _wrap(self._a[_unwrap(key)])

    def __setitem__(self, key, value):
        self._a[_unwrap(key)] = _unwrap(value)

    def __copy__(self):
        return ndarray(self._a.copy())

    def __deepcopy__(self, memo):
        return ndarray(self._a.copy())

    # --- numpy protocols -----------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _strict(inputs, ufunc.__name__)
        out = kwargs.get("out")
        res = getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs))
        if out is not None:
            return out[0] if isinstance(out, tuple) and len(out) == 1 else out
        return _wrap(res)

    def __array_function__(self, func, types_, args, kwargs):
        mine = globals().get(func.__name__)
        if mine is None:
            mine = __getattr__(func.__name__)
        return mine(*args, **kwargs)


def _from_host(a):
    # fresh, aligned allocation like a device array restored from a pickle
    return ndarray(_np.array(a, copy=True, order="K"))


def _wrap(x):
    if isinstance(x, _HOST):
        return ndarray(x)
    if isinstance(x, _np.generic) and not isinstance(
        x, (_np.str_, _np.bytes_, _np.void)
    ):
        return ndarray(_np.asarray(x))
    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)
    if isinstance(x, list):
        return [_wrap(i) for i in x]
    return x


def _unwrap(x):
    if isinstance(x, ndarray):
        return x._a
    if isinstance(x, tuple):
        return tuple(_unwrap(i) for i in x)
    if isinstance(x, list):
        return [_unwrap(i) for i in x]
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _strict(args, where, first_is_data=False):
    """Raise on NumPy arrays anywhere (top level or inside tuples/lists).

    0-d NumPy arrays pass, like NumPy scalars: CuPy treats them as scalars.
    With `first_is_data`, a list or tuple as first argument raises too.
    """
    for i, a in enumerate(args):
        if isinstance(a, _HOST) and a.ndim > 0:
            raise _err(a, f" in {where}")
        if isinstance(a, (list, tuple)):
            if i == 0 and first_is_data:
                raise _err(a, f" as array argument of {where}")
            _strict(a, where)


# binary / unary operators ------------------------------------------------------
def _binop(op):
    def f(self, other):
        if isinstance(other, _HOST) and other.ndim > 0:
            raise _err(other, f" in operator {op.__name__}")
        return _wrap(op(self._a, _unwrap(other)))

    return f


def _rbinop(op):
    def f(self, other):
        if isinstance(other, _HOST) and other.ndim > 0:
            raise _err(other, f" in operator {op.__name__}")
        return _wrap(op(_unwrap(other), self._a))

    return f


def _unop(op):
    def f(self):
        return _wrap(op(self._a))

    return f


def _ibinop(op):
    def f(self, other):
        if isinstance(other, _HOST) and other.ndim > 0:
            raise _err(other, f" in operator {op.__name__}")
        op(self._a, _unwrap(other))
        return self

    return f


for _name, _op, _iop in (
    ("add", operator.add, operator.iadd),
    ("sub", operator.sub, operator.isub),
    ("mul", operator.mul, operator.imul),
    ("truediv", operator.truediv, operator.itruediv),
    ("floordiv", operator.floordiv, operator.ifloordiv),
    ("mod", operator.mod, operator.imod),
    ("pow", operator.pow, operator.ipow),
    ("matmul", operator.matmul, operator.imatmul),
    ("and", operator.and_, operator.iand),
    ("or", operator.or_, operator.ior),
    ("xor", operator.xor, operator.ixor),
    ("lshift", operator.lshift, operator.ilshift),
    ("rshift", operator.rshift, operator.irshift),
):
    setattr(ndarray, f"__{_name}__", _binop(_op))
    setattr(ndarray, f"__r{_name}__", _rbinop(_op))
    setattr(ndarray, f"__i{_name}__", _ibinop(_iop))
for _name, _op in (
    ("lt", operator.lt),
    ("le", operator.le),
    ("gt", operator.gt),
    ("ge", operator.ge),
    ("eq", operator.eq),
    ("ne", operator.ne),
):
    setattr(ndarray, f"__{_name}__", _binop(_op))
for _name, _op in (
    ("neg", operator.neg),
    ("pos", operator.pos),
    ("abs", operator.abs),
    ("invert", operator.invert),
):
    setattr(ndarray, f"__{_name}__", _unop(_op))


# module functions --------------------------------------------------------------
_CREATION = {
    "array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
    "zeros", "ones", "empty", "full", "arange", "linspace", "logspace", "eye",
    "identity", "indices", "fromfunction", "frombuffer", "diag", "tri",
    "geomspace", "copy",
}  # fmt: skip
_SEQUENCE = {
    "concatenate", "stack", "hstack", "vstack", "dstack", "column_stack",
    "row_stack", "block", "meshgrid", "broadcast_arrays", "ix_",
    "ravel_multi_index", "atleast_1d", "atleast_2d", "atleast_3d", "einsum",
    "result_type",
}  # fmt: skip
# Python-level CuPy routines that start with cupy.asanyarray(a), hence accept
# host arrays (one copy)
_ASANYARRAY = {
    "diff", "ediff1d", "gradient", "trapezoid", "unique", "flip", "rot90",
    "roll", "cumsum", "cumprod",
}  # fmt: skip
_PASSTHROUGH = {
    "dtype", "issubdtype", "promote_types", "can_cast", "finfo", "iinfo",
    "isscalar", "result_type", "broadcast_shapes", "get_printoptions",
    "set_printoptions", "printoptions", "errstate", "seterr", "geterr",
    "iterable", "ndim", "shape",
}  # fmt: skip


def _wrap_callable(f, strict, name, first_is_data=False):
    def w(*args, **kwargs):
        if strict:
            _strict(args, name, first_is_data)
            _strict(tuple(kwargs.values()), name)
        res = f(*_unwrap(args), **_unwrap(kwargs))
        # like CuPy: no new array object when nothing was copied
        # (e.g. ascontiguousarray of a contiguous array)
        for a in args[:1]:
            if isinstance(a, ndarray) and isinstance(res, _HOST):
                same = res is a._a or (
                    res.base is a._a
                    and res.shape == a._a.shape
                    and res.strides == a._a.strides
                    and res.dtype == a._a.dtype
                )
                if same:
                    return a
        return _wrap(res)

    w.__name__ = getattr(f, "__name__", name)
    w.__doc__ = getattr(f, "__doc__", None)
    return w


def _module_attr(name, src=_np, prefix="cupy"):
    attr = getattr(src, name)
    if isinstance(attr, (type, _np.dtype)) or not callable(attr):
        return attr
    if name in _PASSTHROUGH:
        return lambda *a, **k: attr(*_unwrap(a), **_unwrap(k))
    if name in _CREATION or name in _ASANYARRAY:
        return _wrap_callable(attr, strict=False, name=f"{prefix}.{name}")
    if name in _SEQUENCE:
        return _wrap_callable(attr, strict=True, name=f"{prefix}.{name}")
    return _wrap_callable(
        attr, strict=True, name=f"{prefix}.{name}", first_is_data=True
    )


def asarray(a, dtype=None, order=None, **kwargs):
    return ndarray(
        _np.array(
            _unwrap(a), dtype=dtype, order=order or "K", copy=kwargs.pop("copy", None)
        )
    )


def array(a, dtype=None, copy=True, order="K", ndmin=0, **kwargs):
    return ndarray(
        _np.array(_unwrap(a), dtype=dtype, copy=copy, order=order, ndmin=ndmin)
    )


def asnumpy(a, *args, **kwargs):
    return a.get() if isinstance(a, ndarray) else _np.asarray(a)


def is_available():
    return True


def get_array_module(*args):
    if builtins.any(isinstance(a, ndarray) for a in args):
        return _sys.modules["cupy"]
    return _np


def fuse(*args, **kwargs):
    """``cupy.fuse``: a plain decorator here (the function runs unfused)."""
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda f: f


class _Pool:
    def free_all_blocks(self):
        pass

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0


def get_default_memory_pool():
    return _Pool()


def get_default_pinned_memory_pool():
    return _Pool()


class _NoGPU:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("fake CuPy cannot run CUDA kernels")

    def get_function(self, name):
        return _NoGPU()


RawKernel = RawModule = ElementwiseKernel = ReductionKernel = _NoGPU


class _Device:
    def __init__(self, id=0):
        self.id = 0 if id is None else id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def use(self):
        pass

    def synchronize(self):
        pass

    @property
    def attributes(self):
        return {"MaxSharedMemoryPerBlock": 48 * 1024}


class _Event:
    def __init__(self, **kwargs):
        pass

    @property
    def done(self):
        return True

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class _Stream(_Device):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __enter__(self):
        return self

    def record(self, event=None):
        event = _Event() if event is None else event
        event.record(self)
        return event

    def wait_event(self, event):
        pass

    @property
    def done(self):
        return True


_Stream.null = _Stream()


def _alloc_pinned_memory(nbytes):
    raise NotImplementedError("fake CuPy has no pinned memory")


cuda = types.ModuleType("cupy.cuda")
cuda.Device = _Device
cuda.Stream = _Stream
cuda.Event = _Event
cuda.get_current_stream = lambda: _Stream.null
cuda.alloc_pinned_memory = _alloc_pinned_memory
cuda.device = types.ModuleType("cupy.cuda.device")
cuda.device.Device = _Device
cuda.runtime = types.ModuleType("cupy.cuda.runtime")
cuda.runtime.getDevice = lambda: 0
cuda.runtime.setDevice = lambda device: None
cuda.runtime.getDeviceCount = lambda: 1
cuda.runtime.memGetInfo = lambda: (1 << 34, 1 << 34)
cuda.runtime.CUDARuntimeError = RuntimeError
cuda.driver = types.SimpleNamespace(CUDADriverError=RuntimeError)


def _submodule(name, src):
    m = types.ModuleType(f"cupy.{name}")
    m.__getattr__ = lambda attr: _module_attr(attr, src, prefix=f"cupy.{name}")
    m.__all__ = [n for n in dir(src) if not n.startswith("_")]
    return m


linalg = _submodule("linalg", _np.linalg)
fft = _submodule("fft", _np.fft)


class _RNGProxy:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if callable(attr):
            return _wrap_callable(attr, strict=False, name=f"cupy.random.{name}")
        return attr


random = types.ModuleType("cupy.random")
random.default_rng = lambda seed=None: _RNGProxy(_np.random.default_rng(_unwrap(seed)))
random.__getattr__ = lambda attr: _wrap_callable(
    getattr(_np.random, attr), strict=False, name=f"cupy.random.{attr}"
)

for _m in (cuda, cuda.device, cuda.runtime, linalg, fft, random):
    _sys.modules[_m.__name__] = _m

bool_ = _np.bool_
_SKIP = {
    "ndarray", "array", "asarray", "linalg", "fft", "random", "cuda", "testing",
    "ma", "char", "rec", "lib", "polynomial", "ctypeslib", "emath", "typing",
    "exceptions", "dtypes", "strings", "f2py", "version", "matlib",
}  # fmt: skip
__all__ = sorted(  # noqa: PLE0605 - the NumPy namespace plus the CuPy extras
    {n for n in dir(_np) if not n.startswith("_") and n not in _SKIP}
    | {
        "ndarray",
        "array",
        "asarray",
        "asnumpy",
        "cuda",
        "linalg",
        "fft",
        "random",
        "is_available",
        "bool_",
        "fuse",
    }
)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    value = _module_attr(name)
    # never shadow bool, int, all, ... used inside this module
    if not hasattr(builtins, name):
        globals()[name] = value
    return value
