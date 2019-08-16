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


import sys
import os
import platform
import ctypes
import glob

if sys.platform == "linux" or platform == "linux2":
    dll_files = ["libstdc++.so*", "libdepthengine.so*", "libk4a.so*", "libk4arecord.so*"]
    pwd = os.path.dirname(os.path.realpath(__file__))
    for dll_file in dll_files:
        full_paths = glob.glob(pwd + "/" + dll_file)
        if len(full_paths) != 1:
            raise RuntimeError("Not found or more than one libs found for", dll_file)
        else:
            ctypes.cdll.LoadLibrary(full_paths[0])
elif sys.platform == "darwin":
    pass
elif sys.platform == "win32":
    dll_files = ["depthengine*.dll", "k4a.dll", "k4arecord.dll"]
    pwd = os.path.dirname(os.path.realpath(__file__))
    for dll_file in dll_files:
        full_paths = glob.glob(pwd + "/" + dll_file)
        if len(full_paths) != 1:
            raise RuntimeError("Not found or more than one libs found for", dll_file)
        else:
            print("loading full_paths[0]")
            ctypes.cdll.LoadLibrary(full_paths[0])
else:
    raise RuntimeError("Unsupported system " + sys.platform)

from .open3d import * # py2 py3 compatible

__version__ = '@PROJECT_VERSION@'

if "@ENABLE_JUPYTER@" == "ON":
    from open3d.j_visualizer import *

    def _jupyter_nbextension_paths():
        return [{
            'section': 'notebook',
            'src': 'static',
            'dest': 'open3d',
            'require': 'open3d/extension'
        }]
