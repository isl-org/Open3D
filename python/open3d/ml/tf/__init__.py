"""Tensorflow specific machine learning functions."""
import os as _os
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

if _build_config['BUNDLE_OPEN3D_ML']:
    if 'OPEN3D_ML_ROOT' in _os.environ:
        from ml3d import configs
        from ml3d import datasets  # this is for convenience to have everything on the same level.
        from ml3d import utils
        from ml3d.tf import dataloaders
        from ml3d.tf import models
        from ml3d.tf import modules
        from ml3d.tf import pipelines
    else:
        # import from the bundled ml3d module.
        from open3d._ml3d import configs
        from open3d._ml3d import datasets  # this is for convenience to have everything on the same level.
        from open3d._ml3d import utils
        from open3d._ml3d.tf import dataloaders
        from open3d._ml3d.tf import models
        from open3d._ml3d.tf import modules
        from open3d._ml3d.tf import pipelines
