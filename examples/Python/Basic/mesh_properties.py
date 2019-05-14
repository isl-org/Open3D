# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
from open3d import *

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

def mesh_generator():
    yield 'box', create_mesh_box()
    yield 'sphere', create_mesh_sphere()
    yield 'cone', create_mesh_cone()
    yield 'torus', create_mesh_torus(radial_resolution=30, tubular_resolution=20)

    mesh = create_mesh_box()
    mesh.triangles = Vector3iVector(np.asarray(mesh.triangles)[:-2])
    yield 'open box', mesh

    mesh0 = create_mesh_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = create_mesh_box()
    mesh1.transform(T)
    mesh = cat_meshes(mesh0, mesh1)
    yield 'boxes', mesh

def check_properties(name, mesh):
    def fmt_bool(b):
        return 'yes' if b else 'no'
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = edge_manifold and vertex_manifold and not self_intersecting
    # orientable = mesh.is_orientable()
    orientable = True
    print(name)
    print('  edge_manifold:     %s' % fmt_bool(edge_manifold))
    print('  vertex_manifold:   %s' % fmt_bool(vertex_manifold))
    print('  self_intersecting: %s' % fmt_bool(self_intersecting))
    print('  watertight:        %s' % fmt_bool(watertight))
    print('  orientable:        %s' % fmt_bool(orientable))

    mesh.compute_vertex_normals()
    draw_geometries([mesh])

if __name__ == "__main__":
    for name, mesh in mesh_generator():
        check_properties(name, mesh)

