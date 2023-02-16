# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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

import open3d as o3d
import numpy as np
import os

if __name__ == "__main__":

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    print("")
    print(
        "Parameters for o3d.pipelines.registration.PoseGraph optimization ...")
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
    )
    option = o3d.pipelines.registration.GlobalOptimizationOption()
    print("")
    print(method)
    print(criteria)
    print(option)
    print("")

    print(
        "Optimizing Fragment o3d.pipelines.registration.PoseGraph using open3d ..."
    )

    pose_graph_data = o3d.data.DemoPoseGraphOptimization()
    pose_graph_fragment = o3d.io.read_pose_graph(
        pose_graph_data.pose_graph_fragment_path)
    print(pose_graph_fragment)
    o3d.pipelines.registration.global_optimization(pose_graph_fragment, method,
                                                   criteria, option)
    o3d.io.write_pose_graph(
        os.path.join('pose_graph_example_fragment_optimized.json'),
        pose_graph_fragment)
    print("")

    print(
        "Optimizing Global o3d.pipelines.registration.PoseGraph using open3d ..."
    )
    pose_graph_global = o3d.io.read_pose_graph(
        pose_graph_data.pose_graph_global_path)
    print(pose_graph_global)
    o3d.pipelines.registration.global_optimization(pose_graph_global, method,
                                                   criteria, option)
    o3d.io.write_pose_graph(
        os.path.join('pose_graph_example_global_optimized.json'),
        pose_graph_global)
