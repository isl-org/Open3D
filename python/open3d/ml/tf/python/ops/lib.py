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
# A CUDA-enabled wheel bundles both CPU- and CUDA-linked ops (open3d/cpu and
# open3d/cuda). Try the CUDA ops first; if TensorFlow lacks a matching CUDA
# runtime the load fails and we fall back to the CPU ops below. A CPU-only wheel
# ships just the cpu ops.
_lib_arch = ('cuda', 'cpu') if _build_config["BUILD_CUDA_MODULE"] else ('cpu',)
_lib_path.extend([
    _os.path.join(_package_root, _la, 'open3d_tf_ops' + _lib_suffix + _lib_ext)
    for _la in _lib_arch
])

# The ops live in open3d/{cpu,cuda}, one level below the package root that holds
# Open3D.dll. On Windows add the package root to the DLL search path so the
# dependent Open3D.dll is found when TensorFlow loads the ops.
if _sys.platform == 'win32':
    _os.add_dll_directory(_os.path.abspath(_package_root))

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

if not _loaded:
    raise _load_except
