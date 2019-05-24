# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_simplification.py

import numpy as np
import open3d as o3d

import meshes


def create_mesh_plane():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
                 dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [2, 0,
                                                                      3]]))
    return mesh


def mesh_generator():
    mesh = meshes.plane()
    yield o3d.geometry.subdivide_midpoint(mesh, 2)

    mesh = o3d.geometry.create_mesh_box()
    yield o3d.geometry.subdivide_midpoint(mesh, 2)

    mesh = o3d.geometry.create_mesh_sphere()
    yield o3d.geometry.subdivide_midpoint(mesh, 2)

    mesh = o3d.geometry.create_mesh_cone()
    yield o3d.geometry.subdivide_midpoint(mesh, 2)

    mesh = o3d.geometry.create_mesh_cylinder()
    yield o3d.geometry.subdivide_midpoint(mesh, 2)

    yield meshes.bathtub()


if __name__ == "__main__":
    np.random.seed(42)

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.vertices).shape[0]
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.random.uniform(0, 1, size=(n_verts, 3)))

        print("original mesh has %d triangles and %d vertices" % (np.asarray(
            mesh.triangles).shape[0], np.asarray(mesh.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh])

        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 4
        target_number_of_triangles = np.asarray(mesh.triangles).shape[0] // 2

        mesh_smp = o3d.geometry.simplify_vertex_clustering(
            mesh,
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        print(
            "vertex clustered mesh (average) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.triangles).shape[0],
             np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = o3d.geometry.simplify_vertex_clustering(
            mesh,
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Quadric)
        print(
            "vertex clustered mesh (quadric) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.triangles).shape[0],
             np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = o3d.geometry.simplify_quadric_decimation(
            mesh, target_number_of_triangles=target_number_of_triangles)
        print("quadric decimated mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_smp.triangles).shape[0],
               np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])
