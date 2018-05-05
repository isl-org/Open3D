# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import importlib
from open3d import *

globals().update(importlib.import_module('open3d').__dict__)
