# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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

try:
    # Azure Kinect is not officially supported on Ubuntu 16.04, this is an
    # unofficial workaround. Install the fix package with
    # `pip install open3d_azure_kinect_ubuntu1604_fix`
    import open3d_azure_kinect_ubuntu1604_fix
except:
    pass

# Workaround when multiple copies of the OpenMP runtime have been linked to
# the program, which happens when PyTorch loads OpenMP runtime first. Not that
# this method is "unsafe, unsupported, undocumented", but we found it to be
# generally safe to use. This should be deprecated once we found a way to
# "ensure that only a single OpenMP runtime is linked into the process".
#
# https://github.com/llvm-mirror/openmp/blob/8453ca8594e1a5dd8a250c39bf8fcfbfb1760e60/runtime/src/i18n/en_US.txt#L449
# https://github.com/dmlc/xgboost/issues/1715
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from open3d.open3d_pybind import camera
from open3d.open3d_pybind import color_map
from open3d.open3d_pybind import geometry
from open3d.open3d_pybind import integration
from open3d.open3d_pybind import io
from open3d.open3d_pybind import odometry
from open3d.open3d_pybind import registration
from open3d.open3d_pybind import utility
from open3d.open3d_pybind import visualization

from open3d.open3d_pybind import Dtype
from open3d.open3d_pybind import Device
from open3d.open3d_pybind import DtypeUtil
from open3d.open3d_pybind import cuda
from open3d.core import SizeVector
from open3d.core import Tensor
from open3d.core import TensorList

from open3d.open3d_pybind import NoneType
none = NoneType()

__version__ = '@PROJECT_VERSION@'

if "@ENABLE_JUPYTER@" == "ON":
    from .j_visualizer import *

    def _jupyter_nbextension_paths():
        return [{
            'section': 'notebook',
            'src': 'static',
            'dest': 'open3d',
            'require': 'open3d/extension'
        }]


_build_config = {
    "BUILD_TENSORFLOW_OPS": "@BUILD_TENSORFLOW_OPS@" == "ON",
}
