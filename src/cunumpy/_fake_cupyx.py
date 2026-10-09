"""The ``cupyx`` package of the fake CuPy: ``cupyx.scipy`` on SciPy, for tests only.

Installed with the fake ``cupy`` (:func:`cunumpy._fake_cupy.install`), so code
written against ``xp.scipy`` runs its CuPy path without a GPU. Each forwarded
SciPy function, class and method behaves like its CuPy counterpart where it
matters for finding host/device bugs:

* array arguments must be fake device arrays: NumPy arrays raise, as they do
  in CuPy;
* array results are fake device arrays, and objects (a fitted spline, a
  factorization) are proxies whose methods follow the same rules;
* each subpackage has only the names CuPy provides (:data:`NAMES`), so code
  using a SciPy name that CuPy lacks (e.g. ``RectBivariateSpline``,
  ``special.jv``, ``linalg.solve_circulant``) fails here as it would on a
  GPU. A CuPy name that the installed SciPy lacks is missing too.

``cupyx.empty_pinned`` and ``cupyx.zeros_pinned`` return NumPy arrays, like
CuPy's (which are backed by pinned memory).
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np

#: The subpackages of ``cupyx.scipy``; the same as ``_scipy_backend.SUBMODULES``.
SUBPACKAGES = (
    "fft",
    "fftpack",
    "interpolate",
    "linalg",
    "ndimage",
    "signal",
    "sparse",
    "sparse.csgraph",
    "sparse.linalg",
    "spatial",
    "special",
    "stats",
)


def _names(text: str) -> frozenset[str]:
    return frozenset(text.split())


#: The public names of each ``cupyx.scipy`` subpackage (CuPy v14.2.0).
NAMES = {
    "fft": _names(
        """
        dct dctn dst dstn fft fft2 fftfreq fftn fftshift fht get_fft_plan hfft
        hfft2 hfftn idct idctn idst idstn ifft ifft2 ifftn ifftshift ifht ihfft
        ihfft2 ihfftn irfft irfft2 irfftn next_fast_len rfft rfft2 rfftfreq
        rfftn
        """,
    ),
    "fftpack": _names(
        """
        fft fft2 fftn get_fft_plan ifft ifft2 ifftn irfft rfft
        """,
    ),
    "interpolate": _names(
        """
        Akima1DInterpolator BPoly BSpline BarycentricInterpolator
        CloughTocher2DInterpolator CubicHermiteSpline CubicSpline
        InterpolatedUnivariateSpline KroghInterpolator LSQUnivariateSpline
        LinearNDInterpolator NdBSpline NdPPoly NearestNDInterpolator PPoly
        PchipInterpolator RBFInterpolator RegularGridInterpolator
        UnivariateSpline barycentric_interpolate interp1d interpn
        krogh_interpolate make_interp_spline make_lsq_spline pchip
        pchip_interpolate splantider splder
        """,
    ),
    "linalg": _names(
        """
        bandwidth block_diag circulant companion convolution_matrix dft expm
        fiedler fiedler_companion hadamard hankel helmert hilbert khatri_rao
        kron leslie lu lu_factor lu_solve solve_triangular toeplitz
        """,
    ),
    "ndimage": _names(
        """
        affine_transform binary_closing binary_dilation binary_erosion
        binary_fill_holes binary_hit_or_miss binary_opening binary_propagation
        black_tophat center_of_mass convolve convolve1d correlate correlate1d
        distance_transform_edt extrema find_objects fourier_ellipsoid
        fourier_gaussian fourier_shift fourier_uniform gaussian_filter
        gaussian_filter1d gaussian_gradient_magnitude gaussian_laplace
        generate_binary_structure generic_filter generic_filter1d
        generic_gradient_magnitude generic_laplace grey_closing grey_dilation
        grey_erosion grey_opening histogram iterate_structure label
        labeled_comprehension laplace map_coordinates maximum maximum_filter
        maximum_filter1d maximum_position mean median median_filter minimum
        minimum_filter minimum_filter1d minimum_position morphological_gradient
        morphological_laplace percentile_filter prewitt rank_filter rotate shift
        sobel spline_filter spline_filter1d standard_deviation sum sum_labels
        uniform_filter uniform_filter1d value_indices variance white_tophat zoom
        """,
    ),
    "signal": _names(
        """
        BadCoefficients CZT StateSpace TransferFunction ZerosPolesGain ZoomFFT
        abcd_normalize argrelextrema argrelmax argrelmin band_stop_obj bilinear
        bilinear_zpk bode buttap butter buttord cheb1ap cheb1ord cheb2ap
        cheb2ord cheby1 cheby2 check_COLA check_NOLA chirp choose_conv_method
        coherence cont2discrete convolve convolve2d correlate correlate2d
        correlation_lags csd cspline1d cspline1d_eval cspline2d cwt czt
        czt_points dbode decimate deconvolve detrend dfreqresp dimpulse dlsim
        dlti dstep ellip ellipap ellipord fftconvolve filtfilt find_peaks
        findfreqs firls firwin firwin2 freqresp freqs freqs_zpk freqz freqz_sos
        freqz_zpk gammatone gauss_spline gausspulse get_window group_delay
        hilbert hilbert2 iircomb iirdesign iirfilter iirnotch iirpeak impulse
        invres invresz istft kaiser_atten kaiser_beta kaiserord lfilter
        lfilter_zi lfiltic lombscargle lp2bp lp2bp_zpk lp2bs lp2bs_zpk lp2hp
        lp2hp_zpk lp2lp lp2lp_zpk lsim lti max_len_seq medfilt medfilt2d
        minimum_phase morlet morlet2 normalize oaconvolve order_filter
        peak_prominences peak_widths periodogram place_poles qmf qspline1d
        qspline1d_eval qspline2d resample resample_poly residue residuez ricker
        savgol_coeffs savgol_filter sawtooth sepfir2d sos2tf sos2zpk sosfilt
        sosfilt_zi sosfiltfilt sosfreqz spectrogram spline_filter square ss2tf
        ss2zpk step stft sweep_poly symiirorder1 symiirorder2 tf2sos tf2ss
        tf2zpk unique_roots unit_impulse upfirdn vectorstrength welch wiener
        zoom_fft zpk2sos zpk2ss zpk2tf
        """,
    ),
    "sparse": _names(
        """
        SparseEfficiencyWarning SparseWarning block_array block_diag bmat
        coo_array coo_matrix csc_array csc_matrix csr_array csr_matrix dia_array
        dia_matrix diags diags_array eye eye_array find get_index_dtype hstack
        identity issparse isspmatrix isspmatrix_coo isspmatrix_csc
        isspmatrix_csr isspmatrix_dia kron kronsum matrix_transpose permute_dims
        rand random random_array safely_cast_index_arrays sparray spdiags
        spmatrix swapaxes tril triu vstack
        """,
    ),
    "sparse.csgraph": _names(
        """
        connected_components
        """,
    ),
    "sparse.linalg": _names(
        """
        LinearOperator SuperLU aslinearoperator bicgstab cg cgs eigsh factorized
        gmres lobpcg lsmr lsqr matrix_power minres norm spilu splu spsolve
        spsolve_triangular svds
        """,
    ),
    "spatial": _names(
        """
        Delaunay KDTree distance_matrix
        """,
    ),
    "special": _names(
        """
        bdtr bdtrc bdtri beta betainc betaincinv betaln binom boxcox boxcox1p
        btdtr btdtri cbrt chdtr chdtrc chdtri cosdg cosm1 cotdg digamma
        ellipeinc ellipj ellipk ellipkinc ellipkm1 entr erf erfc erfcinv erfcx
        erfinv exp1 exp10 exp2 expi expit expm1 expn exprel fdtr fdtrc fdtri
        gamma gammainc gammaincc gammainccinv gammaincinv gammaln gammasgn gdtr
        gdtrc huber i0 i0e i1 i1e inv_boxcox inv_boxcox1p j0 j1 k0 k0e k1 k1e
        kl_div lambertw log1p log_expit log_ndtr log_softmax loggamma logit
        logsumexp lpmv multigammaln nbdtr nbdtrc nbdtri ndtr ndtri pdtr pdtrc
        pdtri poch polygamma pseudo_huber psi radian rel_entr rgamma round
        shichi sici sinc sindg softmax sph_harm sph_harm_y spherical_yn tandg
        wright_bessel xlog1py xlogy y0 y1 yn zeta zetac
        """,
    ),
    "stats": _names(
        """
        boxcox_llf entropy trim_mean zmap zscore
        """,
    ),
}


class _Proxy:
    """A SciPy object (spline, interpolator, ...) seen as a CuPy object."""

    def __init__(self, obj: Any, cupy: Any) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_cupy", cupy)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if callable(attr):
            return _device_callable(
                attr, self._cupy, f"{type(self._obj).__name__}.{name}"
            )
        return _to_device(attr, self._cupy)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._obj, name, _to_host(value, self._cupy))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        where = f"{type(self._obj).__name__}.__call__"
        return _device_callable(self._obj, self._cupy, where)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<fake cupyx {self._obj!r}>"


class _ProxyClass:
    """A SciPy class seen as a CuPy class: instances are :class:`_Proxy`."""

    def __init__(self, cls: type, cupy: Any, where: str) -> None:
        self._cls = cls
        self._cupy = cupy
        self._where = where
        self.__name__ = cls.__name__
        self.__doc__ = cls.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _device_callable(self._cls, self._cupy, self._where)(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._cls, name)
        if callable(attr):
            return _device_callable(attr, self._cupy, f"{self._where}.{name}")
        return attr

    def __instancecheck__(self, obj: Any) -> bool:
        return isinstance(obj, _Proxy) and isinstance(obj._obj, self._cls)

    def __repr__(self) -> str:
        return f"<fake {self._where}>"


def _is_scipy_object(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "scipy"


def _to_device(value: Any, cupy: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, cupy) for v in value)
    if _is_scipy_object(value) and not isinstance(value, np.ndarray):
        return _Proxy(value, cupy)
    return cupy._wrap(value)


def _to_host(value: Any, cupy: Any) -> Any:
    if isinstance(value, _Proxy):
        return value._obj
    if isinstance(value, (tuple, list)):
        return type(value)(_to_host(v, cupy) for v in value)
    if isinstance(value, dict):
        return {k: _to_host(v, cupy) for k, v in value.items()}
    return cupy._unwrap(value)


def _device_callable(fun: Any, cupy: Any, where: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        cupy._strict(args, where)
        cupy._strict(tuple(kwargs.values()), where)
        out = fun(*_to_host(args, cupy), **_to_host(kwargs, cupy))
        return _to_device(out, cupy)

    call.__name__ = getattr(fun, "__name__", where)
    call.__doc__ = getattr(fun, "__doc__", None)
    return call


def _subpackage(path: str, cupy: Any) -> types.ModuleType:
    module = types.ModuleType(f"cupyx.scipy.{path}")
    allowed = NAMES[path]

    def __getattr__(name: str) -> Any:
        if name.startswith("__") or name not in allowed:
            raise AttributeError(
                f"module 'cupyx.scipy.{path}' has no attribute {name!r}"
            )
        attr = getattr(importlib.import_module(f"scipy.{path}"), name, None)
        if attr is None or isinstance(attr, types.ModuleType):
            raise AttributeError(
                f"module 'cupyx.scipy.{path}' has no attribute {name!r}"
            )
        where = f"cupyx.scipy.{path}.{name}"
        if isinstance(attr, type):
            return _ProxyClass(attr, cupy, where)
        if callable(attr):
            return _device_callable(attr, cupy, where)
        return attr

    module.__getattr__ = __getattr__
    return module


def install(cupy: Any) -> None:
    """Install ``cupyx`` and ``cupyx.scipy.*`` into ``sys.modules``, on the fake `cupy`."""
    cupyx = types.ModuleType("cupyx")
    cupyx.__cunumpy_fake__ = True
    cupyx.empty_pinned = lambda shape, dtype=float, order="C": np.empty(
        shape, dtype, order
    )
    cupyx.zeros_pinned = lambda shape, dtype=float, order="C": np.zeros(
        shape, dtype, order
    )
    scipy = types.ModuleType("cupyx.scipy")
    cupyx.scipy = scipy
    modules = {"cupyx": cupyx, "cupyx.scipy": scipy}
    for path in SUBPACKAGES:
        modules[f"cupyx.scipy.{path}"] = _subpackage(path, cupy)
    # parents expose their subpackages as attributes, like real packages
    for name, module in modules.items():
        parent, _, leaf = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], leaf, module)
    sys.modules.update(modules)
