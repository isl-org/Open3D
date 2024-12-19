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


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)


def execute_global_registration(source, target, source_fpfh, target_fpfh,
                                voxel_size):
    distance_threshold = voxel_size * 1.4
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


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

            result = execute_global_registration(source_down, target_down,
                                                 source_fpfh, target_fpfh,
                                                 voxel_size)
            if (result.transformation.trace() == 4.0):
                success = False
            else:
                success = True

            # Note: we save inverse of result.transformation
            # to comply with http://redwood-data.org/indoor/fileformat.html
            if not success:
                print("No reasonable solution.")
            else:
                alignment.append(
                    (s, t, n_ply_files, np.linalg.inv(result.transformation)))
                print(np.linalg.inv(result.transformation))

            if do_visualization:
                draw_registration_result(source_down, target_down,
                                         result.transformation)
