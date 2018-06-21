# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
import sys

if sys.platform == "linux" or sys.platform == "linux2":
    from open3d.linux import *
elif sys.platform == "darwin":
    from open3d.macos import *
elif sys.platform == "win32":
    from open3d.win32 import *
