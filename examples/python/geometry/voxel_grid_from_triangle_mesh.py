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
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    # Fit to unit cube.
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    print('Displaying input mesh ...')
    o3d.visualization.draw([mesh])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=0.05)
    print('Displaying voxel grid ...')
    o3d.visualization.draw([voxel_grid])
