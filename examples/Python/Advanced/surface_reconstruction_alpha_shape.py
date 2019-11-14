# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/surface_reconstruction_ball_pivoting.py

import open3d as o3d
import numpy as np
import os

import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../Misc"))
import meshes

if __name__ == "__main__":
    mesh = meshes.bunny()
    pcd = mesh.sample_points_poisson_disk(200)
    # o3d.visualization.draw_geometries([pcd])

    alpha = 0.05

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
