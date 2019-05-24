# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
import time

import open3d as o3d


def cat_meshes(mesh0, mesh1):
    mesh = o3d.geometry.TriangleMesh()
    vertices0 = np.asarray(mesh0.vertices)
    vertices = np.vstack((vertices0, np.asarray(mesh1.vertices)))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    triangles = np.vstack((np.asarray(mesh0.triangles),
                           np.asarray(mesh1.triangles) + vertices0.shape[0]))
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def mesh_generator():
    yield 'box', o3d.geometry.create_mesh_box()
    yield 'sphere', o3d.geometry.create_mesh_sphere()
    yield 'cone', o3d.geometry.create_mesh_cone()
    yield 'torus', o3d.geometry.create_mesh_torus(radial_resolution=30,
                                                  tubular_resolution=20)
    yield 'moebius (twists=1)', o3d.geometry.create_mesh_moebius(twists=1)
    yield 'moebius (twists=2)', o3d.geometry.create_mesh_moebius(twists=2)
    yield 'moebius (twists=3)', o3d.geometry.create_mesh_moebius(twists=3)

    yield 'knot', o3d.io.read_triangle_mesh('../../TestData/knot.ply')

    verts = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
                     dtype=np.float64)
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    yield 'non-manifold edge', mesh

    verts = np.array([[-1, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 0],
                      [-1, 0, 1], [1, 0, 1], [0, 1, 1]],
                     dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3], [4, 5, 6],
                          [4, 5, 3], [5, 6, 3], [4, 6, 3]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    yield 'non-manifold vertex', mesh

    mesh = o3d.geometry.create_mesh_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:-2])
    yield 'open box', mesh

    mesh0 = o3d.geometry.create_mesh_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = o3d.geometry.create_mesh_box()
    mesh1.transform(T)
    mesh = cat_meshes(mesh0, mesh1)
    yield 'boxes', mesh


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def check_properties(name, mesh):

    def fmt_bool(b):
        return 'yes' if b else 'no'

    print(name)
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    print('  edge_manifold:          %s' % fmt_bool(edge_manifold))
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    print('  edge_manifold_boundary: %s' % fmt_bool(edge_manifold_boundary))
    vertex_manifold = mesh.is_vertex_manifold()
    print('  vertex_manifold:        %s' % fmt_bool(vertex_manifold))
    self_intersecting = mesh.is_self_intersecting()
    print('  self_intersecting:      %s' % fmt_bool(self_intersecting))
    watertight = edge_manifold_boundary and vertex_manifold and not self_intersecting
    print('  watertight:             %s' % fmt_bool(watertight))
    orientable = mesh.is_orientable()
    print('  orientable:             %s' % fmt_bool(orientable))

    mesh.compute_vertex_normals()

    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        print('  # visualize non-manifold edges (allow_boundary_edges=True)')
        o3d.visualization.draw_geometries(
            [mesh, edges_to_lineset(mesh, edges, (1, 0, 0))])
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        print('  # visualize non-manifold edges (allow_boundary_edges=False)')
        o3d.visualization.draw_geometries(
            [mesh, edges_to_lineset(mesh, edges, (0, 1, 0))])
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        print('  # visualize non-manifold vertices')
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices)[verts])
        pcl.paint_uniform_color((0, 0, 1))
        o3d.visualization.draw_geometries([mesh, pcl])
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print('  # visualize self-intersecting triangles')
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        o3d.visualization.draw_geometries(
            [mesh, edges_to_lineset(mesh, edges, (1, 1, 0))])
    if watertight:
        print('  # visualize watertight mesh')
        o3d.visualization.draw_geometries([mesh])

    if not edge_manifold:
        print('  # Remove non-manifold edges')
        mesh.remove_non_manifold_edges()
        print(
            f'  # Is mesh now edge-manifold: {fmt_bool(mesh.is_edge_manifold())}'
        )
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    # test mesh properties
    print('#' * 80)
    print('Test mesh properties')
    print('#' * 80)
    for name, mesh in mesh_generator():
        check_properties(name, mesh)

    # fix triangle orientation
    print('#' * 80)
    print('Fix triangle orientation')
    print('#' * 80)
    for name, mesh in mesh_generator():
        mesh.compute_vertex_normals()
        triangles = np.asarray(mesh.triangles)
        rnd_idx = np.random.rand(*triangles.shape).argsort(axis=1)
        rnd_idx[0] = (0, 1, 2)
        triangles = np.take_along_axis(triangles, rnd_idx, axis=1)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d.visualization.draw_geometries([mesh])
        sucess = mesh.orient_triangles()
        print('%s orientated: %s' % (name, 'yes' if sucess else 'no'))
        o3d.visualization.draw_geometries([mesh])

    # intersection tests
    print('#' * 80)
    print('Intersection tests')
    print('#' * 80)
    np.random.seed(30)
    bbox = o3d.geometry.create_mesh_box(20, 20, 20).translate((-10, -10, -10))
    meshes = [o3d.geometry.create_mesh_box() for _ in range(20)]
    meshes.append(o3d.geometry.create_mesh_sphere())
    meshes.append(o3d.geometry.create_mesh_cone())
    meshes.append(o3d.geometry.create_mesh_torus())
    dirs = [np.random.uniform(-0.1, 0.1, size=(3,)) for _ in meshes]
    for mesh in meshes:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((0.5, 0.5, 0.5))
        mesh.translate(np.random.uniform(-7.5, 7.5, size=(3,)))
    vis = o3d.visualization.Visualizer()
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
