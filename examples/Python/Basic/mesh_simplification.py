# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_simplification.py

import numpy as np
import open3d as o3d

import meshes


def mesh_generator():
    mesh = meshes.plane()
    yield mesh.subdivide_midpoint(2)

    mesh = o3d.geometry.TriangleMesh.create_box()
    yield mesh.subdivide_midpoint(2)

    mesh = o3d.geometry.TriangleMesh.create_sphere()
    yield mesh.subdivide_midpoint(2)

    mesh = o3d.geometry.TriangleMesh.create_cone()
    yield mesh.subdivide_midpoint(2)

    mesh = o3d.geometry.TriangleMesh.create_cylinder()
    yield mesh.subdivide_midpoint(2)

    yield meshes.bathtub()

    yield meshes.bunny()


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
        print('voxel_size = %f' % voxel_size)

        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        print(
            "vertex clustered mesh (average) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.triangles).shape[0],
             np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Quadric)
        print(
            "vertex clustered mesh (quadric) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.triangles).shape[0],
             np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles)
        print("quadric decimated mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_smp.triangles).shape[0],
               np.asarray(mesh_smp.vertices).shape[0]))
        o3d.visualization.draw_geometries([mesh_smp])
