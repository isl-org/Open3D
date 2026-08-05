# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import copy

if __name__ == "__main__":
    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    print("Displaying original mesh ...")
    o3d.visualization.draw([mesh])

    print("Displaying mesh of only the first half triangles ...")
    mesh_cropped = copy.deepcopy(mesh)
    mesh_cropped.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh_cropped.triangles)[:len(mesh_cropped.triangles) //
                                           2, :])
    mesh_cropped.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh_cropped.triangle_normals)
        [:len(mesh_cropped.triangle_normals) // 2, :])
    print(mesh_cropped.triangles)
    o3d.visualization.draw([mesh_cropped])
