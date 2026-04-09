# Stub file for Pylance/mypy: exposes all numpy symbols so that
# `import cunumpy as xp` followed by `xp.<Tab>` shows numpy completions.
# At runtime the real __init__.py dispatches to numpy or cupy via __getattr__.
from numpy import *
from numpy import __version__, __config__
from . import xp
