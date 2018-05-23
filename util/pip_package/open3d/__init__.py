# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
from sys import platform

if platform == "linux" or platform == "linux2":
    from open3d.linux import *
elif platform == "darwin":
    from open3d.macos import *
elif platform == "win32":
    from open3d.win32 import *
