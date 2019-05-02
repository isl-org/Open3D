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
    mesh.triangles = Vector3iVector(np.array([[0,1,2], [2,3,0]]))
    return mesh

if __name__ == "__main__":
    np.random.seed(42)

    # mesh = create_mesh_plane()
    # mesh = create_mesh_box()
    # mesh = create_mesh_sphere()
    # mesh = create_mesh_cone()
    # mesh = create_mesh_cylinder()
    # mesh.subdivide_midpoint(2)

    mesh = read_triangle_mesh("../../TestData/bathtub_0154.ply")

    mesh.compute_vertex_normals()
    n_verts = np.asarray(mesh.vertices).shape[0]
    mesh.vertex_colors = Vector3dVector(np.random.uniform(0,1, size=(n_verts,3)))

    print("original mesh has %d faces" % np.asarray(mesh.triangles).shape[0])
    draw_geometries([mesh])

    # mesh.simplify_vertex_clustering(500,
    #         contraction=SimplificationContraction.Average)
    # mesh.simplify_vertex_clustering(500,
    #         contraction=SimplificationContraction.Quadric)
    mesh.simplify_quadric_decimation(750)
    print(np.asarray(mesh.vertices).shape)
    print(np.asarray(mesh.triangles).shape)
    print("simplified mesh has %d faces" % np.asarray(mesh.triangles).shape[0])
    draw_geometries([mesh])
