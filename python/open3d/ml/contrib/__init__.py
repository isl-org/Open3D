# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as _open3d
if _open3d.__DEVICE_API__ == 'cuda':
    from open3d.cuda.pybind.ml.contrib import *
elif _open3d.__DEVICE_API__ == 'xpu':
    from open3d.xpu.pybind.ml.contrib import *
else:
    from open3d.cpu.pybind.ml.contrib import *
