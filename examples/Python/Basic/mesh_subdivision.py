# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
import open3d as o3d


def create_mesh_triangle():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([(np.sqrt(8 / 9), 0, -1 / 3),
                  (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                  (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3)],
                 dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    return mesh


def create_mesh_plane():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
                 dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [2, 0,
                                                                      3]]))
    return mesh


def mesh_generator():
    yield create_mesh_triangle()
    yield create_mesh_plane()
    yield o3d.geometry.create_mesh_tetrahedron()
    yield o3d.geometry.create_mesh_box()
    yield o3d.geometry.create_mesh_octahedron()
    yield o3d.geometry.create_mesh_icosahedron()
    yield o3d.geometry.create_mesh_sphere()
    yield o3d.geometry.create_mesh_cone()
    yield o3d.geometry.create_mesh_cylinder()
    yield o3d.io.read_triangle_mesh("../../TestData/knot.ply")
    yield o3d.io.read_triangle_mesh("../../TestData/bathtub_0154.ply")


if __name__ == "__main__":
    np.random.seed(42)

    number_of_iterations = 3

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.vertices).shape[0]
        colors = np.random.uniform(0, 1, size=(n_verts, 3))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        print("original mesh has %d triangles and %d vertices" % (np.asarray(
            mesh.triangles).shape[0], np.asarray(mesh.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh])

        mesh_up = o3d.geometry.subdivide_midpoint(
            mesh, number_of_iterations=number_of_iterations)
        print("midpoint upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.triangles).shape[0],
               np.asarray(mesh_up.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_up])

        mesh_up = o3d.geometry.subdivide_loop(
            mesh, number_of_iterations=number_of_iterations)
        print("loop upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.triangles).shape[0],
               np.asarray(mesh_up.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_up])
