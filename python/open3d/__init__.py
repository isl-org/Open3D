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
except ImportError:
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
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

__DEVICE_API__ = 'cpu'
from open3d._build_config import _build_config
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    from ctypes import CDLL as _CDLL
    from pathlib import Path as _Path
    try:  # StopIteration if cpu version not available
        _CDLL(next((_Path(__file__).parent / 'cpu').glob('pybind*')))
    except StopIteration:
        pass
    try:
        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = _CDLL(
            next((_Path(__file__).parent / 'cuda').glob('pybind*')))
        if _pybind_cuda.open3d_core_cuda_device_count() > 0:
            from open3d.cuda.pybind import (camera, geometry, io, pipelines,
                                            utility, t)
            from open3d.cuda import pybind
            __DEVICE_API__ = 'cuda'
    except OSError:  # CUDA not installed
        pass
    except StopIteration:  # pybind cuda library not available
        pass

if __DEVICE_API__ == 'cpu':
    from open3d.cpu.pybind import (camera, geometry, io, pipelines, utility, t)
    from open3d.cpu import pybind

import open3d.core
import open3d.visualization

__version__ = "@PROJECT_VERSION@"

if int(sys.version_info[0]) < 3:
    raise Exception("Open3D only supports Python 3.")

if "@BUILD_JUPYTER_EXTENSION@" == "ON":
    from .j_visualizer import *

    def _jupyter_nbextension_paths():
        return [{
            "section": "notebook",
            "src": "static",
            "dest": "open3d",
            "require": "open3d/extension",
        }]


# OPEN3D_ML_ROOT points to the root of the Open3D-ML repo.
# If set this will override the integrated Open3D-ML.
if 'OPEN3D_ML_ROOT' in os.environ:
    print('Using external Open3D-ML in {}'.format(os.environ['OPEN3D_ML_ROOT']))
    sys.path.append(os.environ['OPEN3D_ML_ROOT'])
