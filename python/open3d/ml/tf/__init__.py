# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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
"""TensorFlow specific machine learning functions."""
import os as _os
from tensorflow import __version__ as _tf_version
from open3d import _build_config

if not _build_config["Tensorflow_VERSION"]:
    raise Exception('Open3D was not built with TensorFlow support!')

_o3d_tf_version = _build_config["Tensorflow_VERSION"].split('.')
if _tf_version.split('.')[:2] != _o3d_tf_version[:2]:
    _o3d_tf_version[2] = '*'  # Any patch level is OK
    match_tf_ver = '.'.join(_o3d_tf_version)
    raise Exception('Version mismatch: Open3D needs TensorFlow version {}, but'
                    ' version {} is installed!'.format(match_tf_ver,
                                                       _tf_version))

from . import layers
from . import ops

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
