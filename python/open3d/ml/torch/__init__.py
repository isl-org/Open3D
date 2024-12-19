# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Torch specific machine learning functions."""
import os as _os
import sys as _sys
from packaging.version import parse as _verp
import torch as _torch
from open3d import _build_config

if not _build_config["Pytorch_VERSION"]:
    raise Exception('Open3D was not built with PyTorch support!')
_o3d_torch_version = _verp(_build_config["Pytorch_VERSION"])
# Check match with PyTorch version, any patch level is OK
if _verp(_torch.__version__).release[:2] != _o3d_torch_version.release[:2]:
    match_torch_ver = '.'.join(
        str(v) for v in _o3d_torch_version.release[:2] + ('*',))
    raise Exception('Version mismatch: Open3D needs PyTorch version {}, but '
                    'version {} is installed!'.format(match_torch_ver,
                                                      _torch.__version__))

# Precompiled wheels at
# https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2
# have been compiled with '-Xcompiler -fno-gnu-unique' and have an additional
# attribute that we test here. Print a warning if the attribute is missing.
if (_build_config["BUILD_CUDA_MODULE"] and
        not hasattr(_torch, "_TORCH_NVCC_FLAGS") and
        _verp(_torch.__version__) < _verp("1.9.0")):
    print("""
--------------------------------------------------------------------------------

 Using the Open3D PyTorch ops with CUDA 11 and PyTorch version < 1.9 may have
 stability issues!

 We recommend to compile PyTorch from source with compile flags
   '-Xcompiler -fno-gnu-unique'

 or use the PyTorch wheels at
   https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2


 Ignore this message if PyTorch has been compiled with the aforementioned
 flags.

 See https://github.com/isl-org/Open3D/issues/3324 and
 https://github.com/pytorch/pytorch/issues/52663 for more information on this
 problem.

--------------------------------------------------------------------------------
""")

_lib_path = []
# allow overriding the path to the op library with an env var.
if 'OPEN3D_TORCH_OP_LIB' in _os.environ:
    _lib_path.append(_os.environ['OPEN3D_TORCH_OP_LIB'])

_this_dir = _os.path.dirname(__file__)
_package_root = _os.path.join(_this_dir, '..', '..')
_lib_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}[_sys.platform]
_lib_suffix = '_debug' if _build_config['CMAKE_BUILD_TYPE'] == 'Debug' else ''
_lib_arch = ('cpu',)
if _build_config["BUILD_CUDA_MODULE"] and _torch.cuda.is_available():
    if _torch.version.cuda == _build_config["CUDA_VERSION"]:
        _lib_arch = ('cuda', 'cpu')
    else:
        print("Warning: Open3D was built with CUDA {} but"
              "PyTorch was built with CUDA {}. Falling back to CPU for now."
              "Otherwise, install PyTorch with CUDA {}.".format(
                  _build_config["CUDA_VERSION"], _torch.version.cuda,
                  _build_config["CUDA_VERSION"]))
_lib_path.extend([
    _os.path.join(_package_root, la,
                  'open3d_torch_ops' + _lib_suffix + _lib_ext)
    for la in _lib_arch
])

_load_except = None
_loaded = False
for _lp in _lib_path:
    try:
        _torch.ops.load_library(_lp)
        _torch.classes.load_library(_lp)
        _loaded = True
        break
    except Exception as ex:
        _load_except = ex
        if not _os.path.isfile(_lp):
            print('The op library at "{}" was not found. Make sure that '
                  'BUILD_PYTORCH_OPS was enabled.'.format(
                      _os.path.realpath(_lp)))

if not _loaded:
    raise _load_except

from . import layers
from . import ops
from . import classes

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
