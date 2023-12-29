# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    knot_data = o3d.data.KnotMesh()
    print(f"Reading mesh from file: knot.ply stored at {knot_data.path}")
    mesh = o3d.io.read_triangle_mesh(knot_data.path)
    print(mesh)
    print("Saving mesh to file: copy_of_knot.ply")
    o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)
