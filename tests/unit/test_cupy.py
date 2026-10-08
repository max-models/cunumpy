import numpy as np
import pytest

import cunumpy as xp
from cunumpy.kernels import PyccelKernel


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("column_block", [False, True])
@pytest.mark.parametrize(
    "conversion", ["to_numpy", "host_call", "evaluate_on_host", "pyccel_kernel"]
)
def test_host_conversion_preserves_order(order, column_block, conversion):
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    expected = np.array(np.arange(30.0).reshape(5, 6), order=order)
    with xp.use_backend("cupy"):
        device = xp.to_cupy(expected)
        if column_block:
            device = device[:, 1:4]
            expected = expected[:, 1:4]

        def check_host(array):
            assert isinstance(array, np.ndarray)
            np.testing.assert_array_equal(array, expected)
            assert array.flags.f_contiguous == (order == "F")
            assert array.flags.c_contiguous == (order == "C")
            return array

        if conversion == "to_numpy":
            check_host(xp.to_numpy(device))
        else:
            if conversion == "host_call":
                result = xp.host_call(check_host, device)
            elif conversion == "evaluate_on_host":

                class Model:
                    @xp.evaluate_on_host
                    def evaluate(self, array):
                        return check_host(array)

                result = Model().evaluate(device)
            else:
                result = PyccelKernel(check_host)(device)

            assert xp.is_gpu(result)
            check_host(xp.to_numpy(result))


def test_to_cupy_available():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    import cupy as cp

    with xp.use_backend("cupy"):
        arr = np.array([1, 2, 3])
        arr_cp = xp.to_cupy(arr)
        assert isinstance(arr_cp, cp.ndarray)


def test_to_cupy_not_available():
    if xp.cupy_available():
        pytest.skip("CuPy is installed and functional, cannot test missing cupy error")

    with xp.use_backend("cupy"):
        arr = np.array([1, 2, 3])
        with pytest.raises(ImportError):
            xp.to_cupy(arr)


def test_synchronize():
    # Should not crash on any backend
    xp.synchronize()

    with xp.use_backend("numpy"):
        xp.synchronize()

    with xp.use_backend("cupy"):
        xp.synchronize()


def test_xp_array_cupy():
    if not xp.cupy_available():
        pytest.skip("CuPy not installed or not functional")

    import cupy as cp

    with xp.use_backend("cupy"):
        arr = xp.array([1, 2])
        arr *= 2
        assert isinstance(arr, cp.ndarray)
        assert cp.asnumpy(arr).tolist() == [2, 4]
