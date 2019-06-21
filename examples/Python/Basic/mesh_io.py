# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import open3d as o3d
import os

import meshes

if __name__ == "__main__":
    mesh = meshes.bunny()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    n_vertices = vertices.shape[0]
    vertex_colors = np.random.uniform(0, 1, size=(n_vertices, 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    def test_float_array(array_a, array_b, eps=1e-6):
        diff = array_a - array_b
        dist = np.linalg.norm(diff, axis=1)
        return np.all(dist < eps)

    def test_int_array(array_a, array_b):
        diff = array_a - array_b
        return np.all(diff == 0)

    def compare_mesh(mesh):
        success = True
        if not test_float_array(vertices, np.asarray(mesh.vertices)):
            success = False
            print('[WARNING] vertices are not the same')
        if not test_float_array(vertex_normals, np.asarray(
                mesh.vertex_normals)):
            success = False
            print('[WARNING] vertex_normals are not the same')
        if not test_float_array(
                vertex_colors, np.asarray(mesh.vertex_colors), eps=1e-2):
            success = False
            print('[WARNING] vertex_colors are not the same')
        if not test_int_array(triangles, np.asarray(mesh.triangles)):
            success = False
            print('[WARNING] triangles are not the same')
        if success:
            print('[INFO] written and read mesh are equal')

    print('Write ply file')
    o3d.io.write_triangle_mesh('tmp.ply', mesh)
    print('Read ply file')
    mesh_test = o3d.io.read_triangle_mesh('tmp.ply')
    compare_mesh(mesh_test)
    os.remove('tmp.ply')

    print('Write obj file')
    o3d.io.write_triangle_mesh('tmp.obj', mesh)
    print('Read obj file')
    mesh_test = o3d.io.read_triangle_mesh('tmp.obj')
    compare_mesh(mesh_test)
    os.remove('tmp.obj')
