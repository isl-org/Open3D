# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_filtering.py

import numpy as np
import open3d as o3d

import meshes


def test_mesh(noise=0):
    mesh = meshes.knot()
    if noise > 0:
        vertices = np.asarray(mesh.vertices)
        vertices += np.random.uniform(0, noise, size=vertices.shape)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh


if __name__ == '__main__':
    in_mesh = test_mesh()
    o3d.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_sharpen(number_of_iterations=1, strength=1)
    o3d.visualization.draw_geometries([mesh])

    in_mesh = test_mesh(noise=5)
    o3d.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_smooth_simple(number_of_iterations=1)
    o3d.visualization.draw_geometries([mesh])

    o3d.visualization.draw_geometries([mesh])
    mesh = in_mesh.filter_smooth_laplacian(number_of_iterations=100)
    o3d.visualization.draw_geometries([mesh])

    o3d.visualization.draw_geometries([mesh])
    mesh = in_mesh.filter_smooth_taubin(number_of_iterations=100)
    o3d.visualization.draw_geometries([mesh])
