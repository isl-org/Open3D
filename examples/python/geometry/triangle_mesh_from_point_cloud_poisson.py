# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    eagle = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(eagle.path)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, -np.pi / 4, 0))
    pcd.rotate(R, center=(0, 0, 0))
    print('Displaying input pointcloud ...')
    o3d.visualization.draw([pcd])

    print('Running Poisson surface reconstruction ...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print('Displaying reconstructed mesh ...')
    o3d.visualization.draw([mesh])
