# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
from open3d import *

def is_watertight(mesh):
    return (mesh.is_edge_manifold(allow_boundary_edges=False) and
            mesh.is_vertex_manifold() and
            not mesh.is_self_intersecting())

def cat_meshes(mesh0, mesh1):
    mesh = TriangleMesh()
    vertices0 = np.asarray(mesh0.vertices)
    vertices = np.vstack((vertices0,
                          np.asarray(mesh1.vertices)))
    mesh.vertices = Vector3dVector(vertices)
    triangles = np.vstack((np.asarray(mesh0.triangles),
                           np.asarray(mesh1.triangles) + vertices0.shape[0]))
    mesh.triangles = Vector3iVector(triangles)
    return mesh

if __name__ == "__main__":
    mesh = create_mesh_box()
    print('box is %swatertight' % '' if is_watertight(mesh) else 'NOT ')
    draw_geometries([mesh])

    mesh = create_mesh_torus(radial_resolution=30, tubular_resolution=20)
    print('torus is %swatertight' % '' if is_watertight(mesh) else 'NOT ')
    draw_geometries([mesh])

    mesh = read_triangle_mesh("../../TestData/knot.ply")
    print('knot is %swatertight' % '' if is_watertight(mesh) else 'NOT ')
    draw_geometries([mesh])

    mesh0 = create_mesh_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = create_mesh_box()
    mesh1.transform(T)
    mesh = cat_meshes(mesh0, mesh1)
    print('boxes are %swatertight' % ('' if is_watertight(mesh) else 'NOT '))
    draw_geometries([mesh])

    mesh = create_mesh_box()
    mesh.triangles = Vector3iVector(np.asarray(mesh.triangles)[:-2])
    print('open box is %swatertight' % ('' if is_watertight(mesh) else 'NOT '))
    draw_geometries([mesh])

