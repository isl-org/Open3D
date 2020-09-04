from tensorflow import __version__ as _tf_version
from open3d import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise Exception('Open3D was not built with Tensorflow support!')

_o3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if _tf_version.split('.')[:2] != _o3d_tf_version[:2]:
    _o3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(_o3d_tf_version)
    raise Exception('Version mismatch: Open3D needs Tensorflow version {}, but'
                    ' version {} is installed!'.format(match_tf_ver,
                                                       _tf_version))

from . import layers
from . import ops
