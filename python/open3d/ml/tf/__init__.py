# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""TensorFlow specific machine learning functions."""

import os as _os
from tensorflow import __version__ as _tf_version
from open3d import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise ImportError('Open3D was not built with TensorFlow support!')

_o3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if _tf_version.split('.')[:2] != _o3d_tf_version[:2]:
    _o3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(str(v) for v in _o3d_tf_version)
    raise Exception('Version mismatch: Open3D needs TensorFlow version {}, but'
                    ' version {} is installed!'.format(match_tf_ver,
                                                       _tf_version))

from . import layers
from . import ops

# put contrib at the same level
from open3d.ml import contrib

# The framework independent modules (configs, datasets, vis) and the framework
# specific modules from Open3D-ML pull in the Open3D-ML dependencies (the `ml`
# extra), so they are imported on first use to keep the ops and layers usable
# without them.
_LAZY_SUBMODULES = ("configs", "datasets", "vis", "models", "modules",
                    "pipelines", "dataloaders")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
