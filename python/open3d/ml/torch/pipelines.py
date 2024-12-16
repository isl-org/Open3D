# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""3D ML pipelines for PyTorch."""

import os as _os
from open3d import _build_config

if _build_config['BUNDLE_OPEN3D_ML']:
    if 'OPEN3D_ML_ROOT' in _os.environ:
        from ml3d.torch.pipelines import *
    else:
        from open3d._ml3d.torch.pipelines import *
