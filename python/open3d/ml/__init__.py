# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from open3d.pybind.ml import *

# These submodules pull in Open3D-ML and its dependencies (the `ml` extra), so
# they are imported on first use to keep `import open3d` working without it.
_LAZY_SUBMODULES = ("configs", "datasets", "vis", "utils")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
