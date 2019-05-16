# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_filtering.py

import numpy as np
from open3d import *

def test_mesh(noise=0):
    mesh = read_triangle_mesh('../../TestData/knot.ply')
    if noise > 0:
        vertices = np.asarray(mesh.vertices)
        vertices += np.random.uniform(0, noise, size=vertices.shape)
        mesh.vertices = Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh

if __name__ == '__main__':
    mesh = test_mesh()
    draw_geometries([mesh])

    mesh = test_mesh()
    mesh.filter_sharpen(number_of_iterations=1, strength=1)
    draw_geometries([mesh])

    mesh = test_mesh(noise=5)
    draw_geometries([mesh])
    mesh.filter_smooth_simple(number_of_iterations=1)
    draw_geometries([mesh])

    mesh = test_mesh(noise=5)
    draw_geometries([mesh])
    mesh.filter_smooth_laplacian(number_of_iterations=100)
    draw_geometries([mesh])

    mesh = test_mesh(noise=5)
    draw_geometries([mesh])
    mesh.filter_smooth_taubin(number_of_iterations=100)
    draw_geometries([mesh])
