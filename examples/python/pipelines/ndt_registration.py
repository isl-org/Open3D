# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Normal Distributions Transform registration example."""

import copy
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


if __name__ == "__main__":
    pcd_data = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(pcd_data.paths[0])
    target = o3d.io.read_point_cloud(pcd_data.paths[1])

    trans_init = np.eye(4)

    print("Initial alignment")
    print(
        o3d.pipelines.registration.evaluate_registration(
            source, target, 0.02, trans_init), "\n")

    source_downsampled = source.voxel_down_sample(voxel_size=0.04)

    option = o3d.pipelines.registration.NormalDistributionsTransformOption(
        voxel_size=0.5,
        min_points_per_voxel=6,
        covariance_regularization=1e-6,
        transformation_epsilon=1e-6,
        relative_objective=1e-6,
        max_iteration=200,
        outlier_threshold=9.0,
        neighbor_search_type=1)

    print("Apply Normal Distributions Transform registration")
    reg_ndt = o3d.pipelines.registration.registration_ndt(
        source_downsampled, target, option, trans_init)
    print(reg_ndt)
    print("Transformation is:")
    print(reg_ndt.transformation, "\n")
    draw_registration_result(source, target, reg_ndt.transformation)
