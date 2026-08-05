# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    print("Before Subdivision: ", mesh)
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    mesh = mesh.subdivide_loop(number_of_iterations=1)
    print("After Subdivision: ", mesh)
    print("Displaying subdivided mesh ...")
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
