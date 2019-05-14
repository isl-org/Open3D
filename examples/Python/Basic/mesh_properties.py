# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
import time

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
    yield 'moebius (twists=1)', create_mesh_moebius(twists=1)
    yield 'moebius (twists=2)', create_mesh_moebius(twists=2)
    yield 'moebius (twists=3)', create_mesh_moebius(twists=3)

    yield 'knot', read_triangle_mesh('../../TestData/knot.ply')

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
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = edge_manifold_boundary and vertex_manifold and not self_intersecting
    orientable = mesh.is_orientable()
    print(name)
    print('  edge_manifold:          %s' % fmt_bool(edge_manifold))
    print('  edge_manifold_boundary: %s' % fmt_bool(edge_manifold_boundary))
    print('  vertex_manifold:        %s' % fmt_bool(vertex_manifold))
    print('  self_intersecting:      %s' % fmt_bool(self_intersecting))
    print('  watertight:             %s' % fmt_bool(watertight))
    print('  orientable:             %s' % fmt_bool(orientable))

    mesh.compute_vertex_normals()
    draw_geometries([mesh])

if __name__ == "__main__":
    # test mesh properties
    for name, mesh in mesh_generator():
        check_properties(name, mesh)

    # fix triangle orientation
    for name, mesh in mesh_generator():
        mesh.compute_vertex_normals()
        triangles = np.asarray(mesh.triangles)
        rnd_idx = np.random.rand(*triangles.shape).argsort(axis=1)
        rnd_idx[0] = (0, 1, 2)
        triangles = np.take_along_axis(triangles, rnd_idx, axis=1)
        mesh.triangles = Vector3iVector(triangles)
        draw_geometries([mesh])
        sucess = mesh.orient_triangles()
        print('%s orientated: %s' % (name, 'yes' if sucess else 'no'))
        draw_geometries([mesh])

    # intersection tests
    np.random.seed(30)
    bbox = create_mesh_box(20,20,20).translate((-10,-10,-10))
    meshes = [create_mesh_box() for _ in range(20)]
    meshes.append(create_mesh_sphere())
    meshes.append(create_mesh_cone())
    meshes.append(create_mesh_torus())
    dirs = [np.random.uniform(-0.1, 0.1, size=(3,)) for _ in meshes]
    for mesh in meshes:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((0.5, 0.5, 0.5))
        mesh.translate(np.random.uniform(-7.5, 7.5, size=(3,)))
    vis = Visualizer()
    vis.create_window()
    for mesh in meshes:
        vis.add_geometry(mesh)
    for iter in range(1000):
        for mesh, dir in zip(meshes, dirs):
            mesh.paint_uniform_color((0.5, 0.5, 0.5))
            mesh.translate(dir)
        for idx0, mesh0 in enumerate(meshes):
            collision = False
            collision = collision or mesh0.is_intersecting(bbox)
            for idx1, mesh1 in enumerate(meshes):
                if collision: break
                if idx0 == idx1: continue
                collision = collision or mesh0.is_intersecting(mesh1)
            if collision:
                mesh0.paint_uniform_color((1, 0, 0))
                dirs[idx0] *= -1
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
    vis.destroy_window()
