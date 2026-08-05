# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    bunny = o3d.data.BunnyMesh()
    mesh_in = o3d.io.read_triangle_mesh(bunny.path)
    mesh_in.compute_vertex_normals()

    print("Before Simplification: ", mesh_in)
    o3d.visualization.draw_geometries([mesh_in])

    mesh_smp = mesh_in.simplify_quadric_decimation(
        target_number_of_triangles=6500)
    print("After Simplification target number of triangles = 6500:\n", mesh_smp)
    o3d.visualization.draw_geometries([mesh_smp])

    mesh_smp = mesh_in.simplify_quadric_decimation(
        target_number_of_triangles=1700)
    print("After Simplification target number of triangles = 1700:\n", mesh_smp)
    o3d.visualization.draw_geometries([mesh_smp])
