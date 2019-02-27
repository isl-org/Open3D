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

import importlib
from .open3d import * # py2 py3 compatible
from .open3d.camera import *
from .open3d.geometry import *
from .open3d.odometry import *
from .open3d.registration import *
from .open3d.integration import *
from .open3d.utility import *
from .open3d.visualization import *

globals().update(importlib.import_module('open3d.open3d.camera').__dict__)
globals().update(importlib.import_module('open3d.open3d.geometry').__dict__)
globals().update(importlib.import_module('open3d.open3d.odometry').__dict__)
globals().update(importlib.import_module('open3d.open3d.registration').__dict__)
globals().update(importlib.import_module('open3d.open3d.integration').__dict__)
globals().update(importlib.import_module('open3d.open3d.utility').__dict__)
globals().update(importlib.import_module('open3d.open3d.visualization').__dict__)

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
