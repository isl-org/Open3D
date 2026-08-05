# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(3000)
    print("Displaying input pointcloud ...")
    o3d.visualization.draw([pcd], point_size=5)

    radii = [0.005, 0.01, 0.02, 0.04]
    print('Running ball pivoting surface reconstruction ...')
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw([rec_mesh])
