# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    print("Displaying mesh without normals ...")
    # Invalidate existing normals.
    mesh.triangle_normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    print("normals: \n", np.asarray(mesh.triangle_normals))
    o3d.visualization.draw([mesh])

    print("Computing normals and rendering it ...")
    mesh.compute_vertex_normals()
    print("normals: \n", np.asarray(mesh.triangle_normals))
    o3d.visualization.draw([mesh])
