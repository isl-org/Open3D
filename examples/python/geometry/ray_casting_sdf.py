# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys

if __name__ == "__main__":
    # Load mesh and convert to open3d.t.geometry.TriangleMesh .
    armadillo_data = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh.
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    min_bound = mesh.vertex.positions.min(0).numpy()
    max_bound = mesh.vertex.positions.max(0).numpy()

    xyz_range = np.linspace(min_bound, max_bound, num=64)

    # Query_points is a [64,64,64,3] array.
    query_points = np.stack(np.meshgrid(*xyz_range.T),
                            axis=-1).astype(np.float32)

    # Signed distance is a [64,64,64] array.
    signed_distance = scene.compute_signed_distance(query_points)

    # We can visualize the slices of the distance field directly with matplotlib.
    fig = plt.figure()
    print("Visualizing sdf at each point for the armadillo mesh ...")

    def show_slices(i=int):
        print(f"Displaying slice no.: {i}")
        if i >= 64:
            sys.exit()
        plt.imshow(signed_distance.numpy()[:, :, i % 64])

    animator = anim.FuncAnimation(fig, show_slices, interval=100)
    plt.show()
