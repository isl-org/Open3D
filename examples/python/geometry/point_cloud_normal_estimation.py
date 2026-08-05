# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(5000)
    # Invalidate existing normals.
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    print("Displaying input pointcloud ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    pcd.estimate_normals()
    print("Displaying pointcloud with normals ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    print("Printing the normal vectors ...")
    print(np.asarray(pcd.normals))
