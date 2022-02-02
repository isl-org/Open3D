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
from ctypes import CDLL as _CDLL
from ctypes.util import find_library as _find_library
from pathlib import Path as _Path
import warnings

from open3d._build_config import _build_config
if _build_config["BUILD_GUI"] and not (_find_library('c++abi') or
                                       _find_library('c++')):
    try:  # Preload libc++.so and libc++abi.so (required by filament)
        _CDLL(str(next((_Path(__file__).parent).glob('*c++abi.*'))))
        _CDLL(str(next((_Path(__file__).parent).glob('*c++.*'))))
    except StopIteration:  # Not found: check system paths while loading
        pass

__DEVICE_API__ = 'cpu'
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    try:  # StopIteration if cpu version not available
        _CDLL(str(next((_Path(__file__).parent / 'cpu').glob('pybind*'))))
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CPU Python "
            "bindings were not found. Open3D will not work on systems without"
            " CUDA devices.", ImportWarning)
    try:
        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = _CDLL(
            str(next((_Path(__file__).parent / 'cuda').glob('pybind*'))))
        if _pybind_cuda.open3d_core_cuda_device_count() > 0:
            from open3d.cuda.pybind import (camera, data, geometry, io,
                                            pipelines, utility, t)
            from open3d.cuda import pybind
            __DEVICE_API__ = 'cuda'
        else:
            warnings.warn(
                "Open3D was built with CUDA support, but no suitable CUDA "
                "devices found. If your system has CUDA devices, check your "
                "CUDA drivers and runtime.", ImportWarning)
    except OSError:
        warnings.warn(
            "Open3D was built with CUDA support, but CUDA libraries could "
            "not be found! Check your CUDA installation. Falling back to the "
            "CPU pybind library.", ImportWarning)
    except StopIteration:
        warnings.warn(
            "Open3D was built with CUDA support, but Open3D CUDA Python "
            "binding library not found! Falling back to the CPU Python "
            "binding library.", ImportWarning)

if __DEVICE_API__ == 'cpu':
    from open3d.cpu.pybind import (camera, data, geometry, io, pipelines,
                                   utility, t)
    from open3d.cpu import pybind

import open3d.core
import open3d.visualization

__version__ = "@PROJECT_VERSION@"

if int(sys.version_info[0]) < 3:
    raise Exception("Open3D only supports Python 3.")

if _build_config["BUILD_JUPYTER_EXTENSION"]:
    import platform
    if not (platform.machine().startswith("arm") or
            platform.machine().startswith("aarch")):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                print("Jupyter environment detected. "
                      "Enabling Open3D WebVisualizer.")
                # Set default window system.
                open3d.visualization.webrtc_server.enable_webrtc()
                # HTTP handshake server is needed when Open3D is serving the
                # visualizer webpage. Disable since Jupyter is serving.
                open3d.visualization.webrtc_server.disable_http_handshake()
        except NameError:
            pass
    else:
        warnings.warn("Open3D WebVisualizer is not supported on ARM for now.",
                      RuntimeWarning)

# OPEN3D_ML_ROOT points to the root of the Open3D-ML repo.
# If set this will override the integrated Open3D-ML.
if 'OPEN3D_ML_ROOT' in os.environ:
    print('Using external Open3D-ML in {}'.format(os.environ['OPEN3D_ML_ROOT']))
    sys.path.append(os.environ['OPEN3D_ML_ROOT'])
import open3d.ml


def _jupyter_labextension_paths():
    """Called by Jupyter Lab Server to detect if it is a valid labextension and
    to install the widget.

    Returns:
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Lab copies from this directory
            during widget installation.
        dest: Destination directory name to install widget files to. Jupyter Lab
            copies from `src` directory into <jupyter path>/labextensions/<dest>
            directory during widget installation.
    """
    return [{
        'src': 'labextension',
        'dest': 'open3d',
    }]


def _jupyter_nbextension_paths():
    """Called by Jupyter Notebook Server to detect if it is a valid nbextension
    and to install the widget.

    Returns:
        section: The section of the Jupyter Notebook Server to change.
            Must be 'notebook' for widget extensions.
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Notebook copies from this
            directory during widget installation.
        dest: Destination directory name to install widget files to. Jupyter
            Notebook copies from `src` directory into
            <jupyter path>/nbextensions/<dest> directory during widget
            installation.
        require: Path to importable AMD Javascript module inside the
            <jupyter path>/nbextensions/<dest> directory.
    """
    return [{
        'section': 'notebook',
        'src': 'nbextension',
        'dest': 'open3d',
        'require': 'open3d/extension'
    }]
