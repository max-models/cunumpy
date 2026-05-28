# Stub file for Pylance/mypy: exposes all numpy symbols so that
# `import cunumpy as xp` followed by `xp.<Tab>` shows numpy completions.
# At runtime the real __init__.py dispatches to numpy or cupy via __getattr__.
from typing import Any

import numpy as np
from numpy import *
from numpy import __config__, __version__

from . import xp

def to_numpy(array: Any) -> np.ndarray: ...
def to_cupy(array: Any) -> Any: ...
