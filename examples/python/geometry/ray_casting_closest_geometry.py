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
    cube = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_box().translate([-1.2, -1.2, 0]))
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh.create_sphere(0.5).translate([0.7, 0.8, 0]))

    scene = o3d.t.geometry.RaycastingScene()
    # Add triangle meshes and remember ids.
    mesh_ids = {}
    mesh_ids[scene.add_triangles(cube)] = 'cube'
    mesh_ids[scene.add_triangles(sphere)] = 'sphere'

    # Compute range.
    xyz_range = np.linspace([-2, -2, -2], [2, 2, 2], num=64)
    # Query_points is a [64,64,64,3] array.
    query_points = np.stack(np.meshgrid(*xyz_range.T),
                            axis=-1).astype(np.float32)
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    signed_distance = distance
    closest_geometry = closest_points['geometry_ids'].numpy()

    # We can visualize the slices of the distance field and closest geometry directly with matplotlib.
    fig, axes = plt.subplots(1, 2)
    print(
        "Visualizing sdf and closest geometry at each point for a cube and sphere ..."
    )

    def show_slices(i=int):
        print(f"Displaying slice no.: {i}")
        if i >= 64:
            sys.exit()
        axes[0].imshow(signed_distance[:, :, i])
        axes[1].imshow(closest_geometry[:, :, i])

    animator = anim.FuncAnimation(fig, show_slices, interval=100)
    plt.show()
