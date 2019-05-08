# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
from open3d import *

def create_mesh_triangle():
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(
            np.array([(np.sqrt(8/9), 0, -1/3),
                      (-np.sqrt(2/9), np.sqrt(2/3), -1/3),
                      (-np.sqrt(2/9), -np.sqrt(2/3), -1/3)], dtype=np.float32))
    mesh.triangles = Vector3iVector(np.array([[0,1,2]]))
    return mesh

def create_mesh_plane():
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(
            np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0]], dtype=np.float32))
    mesh.triangles = Vector3iVector(np.array([[0,2,1], [2,0,3]]))
    return mesh

def create_mesh_tetrahedron():
    mesh = TriangleMesh()
    verts = np.array([(np.sqrt(8/9), 0, -1/3),
                      (-np.sqrt(2/9), np.sqrt(2/3), -1/3),
                      (-np.sqrt(2/9), -np.sqrt(2/3), -1/3),
                      (0, 0, 1)], dtype=np.float64)
    faces = np.array([(0,2,1), (0,3,2), (0,1,3), (1,2,3)], dtype=np.int32)
    mesh.vertices = Vector3dVector(verts)
    mesh.triangles = Vector3iVector(faces)
    return mesh

def mesh_generator():
    yield create_mesh_triangle()
    yield create_mesh_plane()
    yield create_mesh_tetrahedron()
    yield create_mesh_box()
    yield create_mesh_sphere()
    yield create_mesh_cone()
    yield create_mesh_cylinder()
    yield read_triangle_mesh("../../TestData/knot.ply")
    yield read_triangle_mesh("../../TestData/bathtub_0154.ply")

if __name__ == "__main__":
    np.random.seed(42)

    number_of_iterations = 3

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.vertices).shape[0]
        mesh.vertex_colors = Vector3dVector(np.random.uniform(0,1, size=(n_verts,3)))

        print("original mesh has %d triangles and %d vertices" %
                (np.asarray(mesh.triangles).shape[0],
                    np.asarray(mesh.vertices).shape[0]))
        draw_geometries([mesh])

        mesh_up = subdivide_midpoint(mesh, number_of_iterations=number_of_iterations)
        print("midpoint upsampled mesh has %d triangles and %d vertices" %
                (np.asarray(mesh_up.triangles).shape[0],
                    np.asarray(mesh_up.vertices).shape[0]))
        draw_geometries([mesh_up])

        mesh_up = subdivide_loop(mesh, number_of_iterations=number_of_iterations)
        print("loop upsampled mesh has %d triangles and %d vertices" %
                (np.asarray(mesh_up.triangles).shape[0],
                    np.asarray(mesh_up.vertices).shape[0]))
        draw_geometries([mesh_up])
