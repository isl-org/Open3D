# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
from sys import platform
from sys import path

if platform == "linux" or platform == "linux2":
    path.append("linux")
elif platform == "darwin":
    path.append("macos")
elif platform == "win32":
    path.append("win32")

from open3d import *
