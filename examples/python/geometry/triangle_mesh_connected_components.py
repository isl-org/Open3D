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
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    vert = np.asarray(mesh.vertices)
    min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
    for _ in range(30):
        cube = o3d.geometry.TriangleMesh.create_box()
        cube.scale(0.005, center=cube.get_center())
        cube.translate(
            (
                np.random.uniform(min_vert[0], max_vert[0]),
                np.random.uniform(min_vert[1], max_vert[1]),
                np.random.uniform(min_vert[2], max_vert[2]),
            ),
            relative=False,
        )
        mesh += cube
    mesh.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw([mesh])

    print("Clustering connected triangles ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("Displaying mesh with small clusters removed ...")
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw([mesh_0])
