# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
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
