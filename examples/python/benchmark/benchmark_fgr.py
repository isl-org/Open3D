# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys
import numpy as np
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *

do_visualization = False


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


if __name__ == "__main__":
    # data preparation
    dataset = o3d.data.LivingRoomPointClouds()
    n_ply_files = len(dataset.paths)
    voxel_size = 0.05

    alignment = []
    for s in range(n_ply_files):
        for t in range(s + 1, n_ply_files):
            print("LivingRoomPointClouds:: matching %d-%d" % (s, t))
            source = o3d.io.read_point_cloud(dataset.paths[s])
            target = o3d.io.read_point_cloud(dataset.paths[t])
            source_down, source_fpfh = preprocess_point_cloud(
                source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(
                target, voxel_size)

            result = execute_fast_global_registration(source_down, target_down,
                                                      source_fpfh, target_fpfh,
                                                      voxel_size)
            if (result.transformation.trace() == 4.0):
                success = False
            else:
                success = True

            # Note: we save inverse of result_ransac.transformation
            # to comply with http://redwood-data.org/indoor/fileformat.html
            alignment.append(
                CameraPose([s, t, n_ply_files],
                           np.linalg.inv(result.transformation)))
            print(np.linalg.inv(result.transformation))

            if do_visualization:
                draw_registration_result(source_down, target_down,
                                         result.transformation)
