# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""TensorFlow specific machine learning functions."""

import os as _os
from tensorflow import __version__ as _tf_version
from open3d import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise Exception('Open3D was not built with TensorFlow support!')

_o3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if _tf_version.split('.')[:2] != _o3d_tf_version[:2]:
    _o3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(str(v) for v in _o3d_tf_version)
    raise Exception('Version mismatch: Open3D needs TensorFlow version {}, but'
                    ' version {} is installed!'.format(match_tf_ver,
                                                       _tf_version))

from . import layers
from . import ops

# put framework independent modules here for convenience
from . import configs
from . import datasets
from . import vis

# framework specific modules from open3d-ml
from . import models
from . import modules
from . import pipelines
from . import dataloaders

# put contrib at the same level
from open3d.ml import contrib
