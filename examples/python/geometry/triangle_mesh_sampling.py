# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    print("Displaying input mesh ...")
    o3d.visualization.draw([mesh])

    print("Displaying pointcloud using uniform sampling ...")
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    o3d.visualization.draw([pcd], point_size=5)

    print("Displaying pointcloud using Poisson disk sampling ...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=1000, init_factor=5)
    o3d.visualization.draw([pcd], point_size=5)
