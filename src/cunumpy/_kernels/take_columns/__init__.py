"""The ``take_columns`` kernel: Pyccel and NumPy versions with the same arguments."""

from cunumpy._dispatch import Kernel
from cunumpy._pyccel_cache import PyccelCache

kernel = Kernel.from_folder(
    __name__,
    host_suffix="_pyccel",
    dispatch="arrays",
    compile_host=PyccelCache("cunumpy"),
)
