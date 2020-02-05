"""This module loads the op library."""
import os as _os
import sys as _sys
import tensorflow as _tf

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
