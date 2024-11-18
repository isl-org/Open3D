# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
from packaging.version import parse as _verp
import paddle as _paddle
from open3d import _build_config

if not _build_config["Paddle_VERSION"]:
    raise Exception('Open3D was not built with Paddle support!')
_o3d_paddle_version = _verp(_build_config["Paddle_VERSION"])
# Check match with Paddle version, any patch level is OK
if _verp(_paddle.__version__).release[:2] != _o3d_paddle_version.release[:2]:
    match_paddle_ver = '.'.join(
        str(v) for v in _o3d_paddle_version.release[:2] + ('*',))
    raise Exception('Version mismatch: Open3D needs Paddle version {}, but '
                    'version {} is installed!'.format(match_paddle_ver,
                                                      _paddle.__version__))

_loaded = False
try:
    from . import ops
    _loaded = True
except Exception as e:
    raise e

from . import layers
from . import classes

# put contrib at the same level
from open3d.ml import contrib
