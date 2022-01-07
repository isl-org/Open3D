# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
"""Torch specific machine learning functions."""
import os as _os
import sys as _sys
import torch as _torch
from open3d import _build_config

if not _build_config["Pytorch_VERSION"]:
    raise Exception('Open3D was not built with PyTorch support!')

_o3d_torch_version = _build_config["Pytorch_VERSION"].split('.')
if _torch.__version__.split('.')[:2] != _o3d_torch_version[:2]:
    _o3d_torch_version[2] = '*'  # Any patch level is OK
    match_torch_ver = '.'.join(_o3d_torch_version)
    raise Exception('Version mismatch: Open3D needs PyTorch version {}, but '
                    'version {} is installed!'.format(match_torch_ver,
                                                      _torch.__version__))

# Precompiled wheels at
# https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2
# have been compiled with '-Xcompiler -fno-gnu-unique' and have an additional
# attribute that we test here. Print a warning if the attribute is missing.
if _build_config["BUILD_CUDA_MODULE"] and not hasattr(_torch,
                                                      "_TORCH_NVCC_FLAGS"):
    print("""
--------------------------------------------------------------------------------

 Using the Open3D PyTorch ops with CUDA 11 may have stability issues!

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
