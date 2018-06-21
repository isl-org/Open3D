# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
import platform

if platform.architecture()[0] == '32bit':
    globals().update(importlib.import_module('open3d.win32.32b.open3d').__dict__)
elif platform.architecture()[0] == '64bit':
    globals().update(importlib.import_module('open3d.win32.64b.open3d').__dict__)
