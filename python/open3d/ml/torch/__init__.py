"""This module loads the op library."""
import os as _os
import sys as _sys
import torch as _torch
from open3d import _build_config

_this_dir = _os.path.dirname(__file__)
_package_root = _os.path.join(_this_dir, '..', '..')
_lib_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}[_sys.platform]
_lib_suffix = '_debug' if _build_config['CMAKE_BUILD_TYPE'] == 'Debug' else ''
_lib_path = _os.path.join(_package_root,
                          'open3d_torch_ops' + _lib_suffix + _lib_ext)

# allow overriding the path to the op library with an env var.
if 'OPEN3D_TORCH_OP_LIB' in _os.environ:
    _lib_path = _os.environ['OPEN3D_TORCH_OP_LIB']

try:
    _torch.ops.load_library(_lib_path)
except Exception as ex:
    if not _os.path.isfile(_lib_path):
        print(
            'The op library at "{}" was not found. Make sure that BUILD_PYTORCH_OPS was enabled.'
            .format(_lib_path))
    raise ex

from . import nn

if open3d._build_config['BUNDLE_OPEN3D_ML']:
    if 'OPEN3D_ML_ROOT' in _os.environ:
        from ml3d import configs
        from ml3d import datasets  # this is for convenience to have everything on the same level
        from ml3d import utils
        from ml3d.torch import dataloaders
        from ml3d.torch import models
        from ml3d.torch import modules
        from ml3d.torch import pipelines
    else:
        # import from the bundled ml3d module
        from open3d._ml3d import configs
        from open3d._ml3d import datasets  # this is for convenience to have everything on the same level
        from open3d._ml3d import utils
        from open3d._ml3d.torch import dataloaders
        from open3d._ml3d.torch import models
        from open3d._ml3d.torch import modules
        from open3d._ml3d.torch import pipelines
