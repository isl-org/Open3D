# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Build a KDTree and use it for neighbour search"""

import open3d as o3d
import numpy as np


def radius_search():
    print("Loading pointcloud ...")
    sample_pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the neighbors of 50000th point with distance less than 0.2, and painting them green ..."
    )
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50000], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


def knn_search():
    print("Loading pointcloud ...")
    sample_pcd = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the 2000 nearest neighbors of 50000th point, and painting them red ..."
    )
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[50000], 2000)
    np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


if __name__ == "__main__":
    knn_search()
    radius_search()
