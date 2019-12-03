"""This module loads the op library and defines shortcuts for the op functions.
"""
import os as _os
import sys as _sys
import tensorflow as _tf
from tensorflow.python.framework import ops as _ops

_this_dir = _os.path.dirname(__file__)
_package_root = _os.path.join(_this_dir, '..', '..', '..', '..')
_lib_ext = {'linux': '.so', 'darwin': '.dylib', 'win32': '.dll'}[_sys.platform]
_lib_path = _os.path.join(_package_root, 'open3d_tf_ops' + _lib_ext)

# allow overriding the path to the op library with an env var.
if 'OPEN3D_TF_OP_LIB' in _os.environ:
    _lib_path = _os.environ['OPEN3D_TF_OP_LIB']

try:
    _lib = _tf.load_op_library(_lib_path)
except Exception as ex:
    if not _os.path.isfile(_lib_path):
        print(
            'The op library at "{}" was not found. Make sure that BUILD_TENSORFLOW_OPS was enabled.'
            .format(_lib_path))
    raise ex


# Add shortcuts without the 'open3d_' prefix to this module.
# TODO do not create the shortcuts at runtime but generate this file to
# allow IDEs to lookup the docstring without actually loading the module.
def _add_function_shortcuts():
    for x in dir(_lib):
        if x.startswith('open3d_'):
            attr = getattr(_lib, x)
            if hasattr(attr, '__call__'):
                truncated_fn_name = x[7:]  # removes the 'open3d_' prefix
                setattr(_sys.modules[__name__], truncated_fn_name, attr)


_add_function_shortcuts()
