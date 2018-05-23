# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
from sys import platform
from sys import path

if platform == "linux" or platform == "linux2":
    from open3d.linux import open3d
    globals().update(importlib.import_module('open3d.linux.open3d').__dict__)
elif platform == "darwin":
    from open3d.linmacosux import open3d
    globals().update(importlib.import_module('open3d.macos.open3d').__dict__)
elif platform == "win32":
    from open3d.win32 import open3d
    globals().update(importlib.import_module('open3d.win32.open3d').__dict__)
