# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Functional API with operators.

These are the building blocks for the layers. See The layer API for an easy to
use high level interface.
"""
import os as _os
import sys as _sys
import types as _types
import importlib as _importlib
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util
import paddle as _paddle
from open3d import _build_config

from ..python.ops import *

_lib_path = []
# allow overriding the path to the op library with an env var.
if 'OPEN3D_PADDLE_OP_LIB' in _os.environ:
    _lib_path.append(_os.environ['OPEN3D_PADDLE_OP_LIB'])

_this_dir = _os.path.dirname(__file__)
_package_root = _os.path.join(_this_dir, '..', '..', '..')
_lib_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}[_sys.platform]
_lib_suffix = '_debug' if _build_config['CMAKE_BUILD_TYPE'] == 'Debug' else ''
_lib_arch = ('cpu',)
if _build_config["BUILD_CUDA_MODULE"] and _paddle.device.cuda.device_count(
) >= 1:
    if _paddle.version.cuda() == _build_config["CUDA_VERSION"]:
        _lib_arch = ('cuda', 'cpu')
    else:
        print("Warning: Open3D was built with CUDA {} but"
              "Paddle was built with CUDA {}. Falling back to CPU for now."
              "Otherwise, install Paddle with CUDA {}.".format(
                  _build_config["CUDA_VERSION"], _paddle.version.cuda(),
                  _build_config["CUDA_VERSION"]))
_lib_path.extend([
    _os.path.join(_package_root, la,
                  'open3d_paddle_ops' + _lib_suffix + _lib_ext)
    for la in _lib_arch
])

# only load first lib
_load_lib_path = _lib_path[0]
# load custom op shared library with abs path
_custom_ops = _paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
    _load_lib_path)

try:
    _spec = _importlib_util.spec_from_file_location(__name__, _load_lib_path)
    assert _spec is not None
    _mod = _importlib_util.module_from_spec(_spec)
    assert isinstance(_spec.loader, _importlib_abc.Loader)
    _spec.loader.exec_module(_mod)
except ImportError:
    _mod = _types.ModuleType(__name__)
