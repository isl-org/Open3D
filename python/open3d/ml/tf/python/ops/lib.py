# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""This module loads the op library."""

import os as _os
import sys as _sys
import tensorflow as _tf
from open3d import _build_config

_lib_path = []
# allow overriding the path to the op library with an env var.
if 'OPEN3D_TF_OP_LIB' in _os.environ:
    _lib_path.append(_os.environ['OPEN3D_TF_OP_LIB'])

_this_dir = _os.path.dirname(__file__)
_package_root = _os.path.join(_this_dir, '..', '..', '..', '..')
_lib_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}[_sys.platform]
_lib_suffix = '_debug' if _build_config['CMAKE_BUILD_TYPE'] == 'Debug' else ''
# CUDA wheels ship open3d/{cpu,cuda} ops; try cuda when torch's CUDA matches
# the wheel, else cpu. CPU-only wheels use cpu ops only.
_lib_arch = ('cuda', 'cpu') if _build_config["BUILD_CUDA_MODULE"] else ('cpu',)
_lib_path.extend([
    _os.path.join(_package_root, _la, 'open3d_tf_ops' + _lib_suffix + _lib_ext)
    for _la in _lib_arch
])

# Ops live in open3d/{cpu,cuda}; on Windows add the package root so Open3D.dll
# (beside this package) is found when torch loads the ops.
_dll_dir = None
if _sys.platform == 'win32':
    _dll_dir = _os.add_dll_directory(_os.path.abspath(_package_root))

_load_except = None
_loaded = False
for _lp in _lib_path:
    try:
        _lib = _tf.load_op_library(_lp)
        _loaded = True
        break
    except Exception as ex:
        _load_except = ex
        if not _os.path.isfile(_lp):
            print('The op library at "{}" was not found. Make sure that '
                  'BUILD_TENSORFLOW_OPS was enabled.'.format(
                      _os.path.realpath(_lp)))

if _dll_dir:
    _dll_dir.close()

if not _loaded:
    raise _load_except
