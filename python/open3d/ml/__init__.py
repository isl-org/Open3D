# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os as _os
import open3d as _open3d
from open3d import _build_config
if _open3d.__DEVICE_API__ == 'cuda':
    from open3d.cuda.pybind.ml import *
else:
    from open3d.cpu.pybind.ml import *

from . import configs
from . import datasets
from . import vis
from . import utils
if _build_config["BUILD_TENSORFLOW_OPS"]:
    from . import tf
if _build_config["BUILD_PYTORCH_OPS"]:
    from . import torch
