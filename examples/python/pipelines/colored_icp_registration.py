# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""ICP variant that uses both geometry and color for registration"""

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target])


print("Load two point clouds and show initial pose ...")
ply_data = o3d.data.DemoColoredICPPointClouds()
source = o3d.io.read_point_cloud(ply_data.paths[0])
target = o3d.io.read_point_cloud(ply_data.paths[1])

if __name__ == "__main__":
    # Draw initial alignment.
    current_transformation = np.identity(4)
    # draw_registration_result(source, target, current_transformation)
    print(current_transformation)

    # Colored pointcloud registration.
    # This is implementation of following paper:
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017.
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("Colored point cloud registration ...\n")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("2. Estimate normal")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp, "\n")
    # draw_registration_result(source, target, result_icp.transformation)
    print(current_transformation)
