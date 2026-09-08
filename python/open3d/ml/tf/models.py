# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""TensorFlow network models."""

import os as _os
from open3d import _build_config
from open3d._optional_deps import require_ml_extra

if _build_config['BUNDLE_OPEN3D_ML']:
    require_ml_extra()
    if 'OPEN3D_ML_ROOT' in _os.environ:
        from ml3d.tf.models import *
    else:
        from open3d._ml3d.tf.models import *
