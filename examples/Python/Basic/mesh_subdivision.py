# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_subdivision.py

import numpy as np
import open3d as o3d

import meshes


def mesh_generator():
    yield meshes.triangle()
    yield meshes.plane()
    yield o3d.geometry.TriangleMesh.create_tetrahedron()
    yield o3d.geometry.TriangleMesh.create_box()
    yield o3d.geometry.TriangleMesh.create_octahedron()
    yield o3d.geometry.TriangleMesh.create_icosahedron()
    yield o3d.geometry.TriangleMesh.create_sphere()
    yield o3d.geometry.TriangleMesh.create_cone()
    yield o3d.geometry.TriangleMesh.create_cylinder()
    yield meshes.knot()
    yield meshes.bathtub()


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

        mesh_up = mesh.subdivide_midpoint(
            number_of_iterations=number_of_iterations)
        print("midpoint upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.triangles).shape[0],
               np.asarray(mesh_up.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_up])

        mesh_up = mesh.subdivide_loop(number_of_iterations=number_of_iterations)
        print("loop upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.triangles).shape[0],
               np.asarray(mesh_up.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_up])
