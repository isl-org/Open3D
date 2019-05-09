# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
from open3d import *

def create_mesh_plane():
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(
            np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0]], dtype=np.float32))
    mesh.triangles = Vector3iVector(np.array([[0,2,1], [2,0,3]]))
    return mesh

def mesh_generator():
    mesh = create_mesh_plane()
    yield subdivide_midpoint(mesh, 2)

    mesh = create_mesh_box()
    yield subdivide_midpoint(mesh, 2)

    mesh = create_mesh_sphere()
    yield subdivide_midpoint(mesh, 2)

    mesh = create_mesh_cone()
    yield subdivide_midpoint(mesh, 2)

    mesh = create_mesh_cylinder()
    yield subdivide_midpoint(mesh, 2)

    mesh = read_triangle_mesh("../../TestData/bathtub_0154.ply")
    yield mesh

if __name__ == "__main__":
    np.random.seed(42)

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.vertices).shape[0]
        mesh.vertex_colors = Vector3dVector(np.random.uniform(0,1, size=(n_verts,3)))

        print("original mesh has %d triangles and %d vertices" %
                (np.asarray(mesh.triangles).shape[0],
                    np.asarray(mesh.vertices).shape[0]))
        draw_geometries([mesh])

        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 4
        target_number_of_triangles = np.asarray(mesh.triangles).shape[0] // 2

        mesh_smp = simplify_vertex_clustering(mesh, voxel_size=voxel_size,
                contraction=SimplificationContraction.Average)
        print("vertex clustered mesh (average) has %d triangles and %d vertices" %
                (np.asarray(mesh_smp.triangles).shape[0],
                    np.asarray(mesh_smp.vertices).shape[0]))
        draw_geometries([mesh_smp])

        mesh_smp = simplify_vertex_clustering(mesh, voxel_size=voxel_size,
                contraction=SimplificationContraction.Quadric)
        print("vertex clustered mesh (quadric) has %d triangles and %d vertices" %
                (np.asarray(mesh_smp.triangles).shape[0],
                    np.asarray(mesh_smp.vertices).shape[0]))
        draw_geometries([mesh_smp])

        mesh_smp = simplify_quadric_decimation(mesh,
                target_number_of_triangles=target_number_of_triangles)
        print("quadric decimated mesh has %d triangles and %d vertices" %
                (np.asarray(mesh_smp.triangles).shape[0],
                    np.asarray(mesh_smp.vertices).shape[0]))
        draw_geometries([mesh_smp])
