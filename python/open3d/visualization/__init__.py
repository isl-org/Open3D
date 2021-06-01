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

import logging as _log
from ctypes import util as _util
from ctypes import CDLL as _cdll
_EGL_AVAILABLE = False
try:
    _egl_name = _util.find_library("EGL")
    if _egl_name is not None:
        _egl = _cdll(_egl_name)
        _EGL_AVAILABLE = bool(_egl.eglInitialize(_egl.eglGetDisplay()))
except OSError:  # Broken OpenGL
    pass

if not _EGL_AVAILABLE:
    _log.debug("EGL not available. Please check your graphics setup.")

import open3d
if open3d.__DEVICE_API__ == 'cuda':
    if "@BUILD_GUI@" == "ON":
        from open3d.cuda.pybind.visualization import gui
    from open3d.cuda.pybind.visualization import *
else:
    if "@BUILD_GUI@" == "ON":
        from open3d.cpu.pybind.visualization import gui
    from open3d.cpu.pybind.visualization import *

from ._external_visualizer import *

if "@BUILD_GUI@" == "ON":
    from .draw import draw
