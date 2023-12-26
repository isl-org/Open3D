# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load two aligned point clouds.")
    demo_data = o3d.data.DemoFeatureMatchingPointClouds()
    pcd0 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd0, pcd1])
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.fpfh_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.fpfh_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")

    print("Load their L32D feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.l32d_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.l32d_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")
