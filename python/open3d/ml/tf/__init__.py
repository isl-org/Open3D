from tensorflow import __version__ as tf_version
from open3d import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise Exception('Open3D was not built with Tensorflow support!')

o3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if tf_version.split('.')[:2] != o3d_tf_version[:2]:
    o3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(o3d_tf_version)
    raise Exception('Version mismatch: Open3D needs Tensorflow version {}, but'
                    ' version {} is installed!'.format(match_tf_ver,
                                                       tf_version))

from . import layers
from . import ops
