# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d
if open3d.__DEVICE_API__ == "cuda":
    if open3d._build_config["BUILD_GUI"]:
        from open3d.cuda.pybind.visualization import gui
    from open3d.cuda.pybind.visualization import *
else:
    if open3d._build_config["BUILD_GUI"]:
        from open3d.cpu.pybind.visualization import gui
    from open3d.cpu.pybind.visualization import *

from ._external_visualizer import *
from .draw_plotly import draw_plotly
from .draw_plotly import draw_plotly_server
from .to_mitsuba import to_mitsuba

if open3d._build_config["BUILD_GUI"]:
    from .draw import draw
