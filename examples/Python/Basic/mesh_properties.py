# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
import open3d as o3d
import time

import meshes


def mesh_generator(edge_cases=True):
    yield 'box', o3d.geometry.TriangleMesh.create_box()
    yield 'sphere', o3d.geometry.TriangleMesh.create_sphere()
    yield 'cone', o3d.geometry.TriangleMesh.create_cone()
    yield 'torus', o3d.geometry.TriangleMesh.create_torus(radial_resolution=30,
                                                          tubular_resolution=20)
    yield 'moebius (twists=1)', o3d.geometry.TriangleMesh.create_moebius(
        twists=1)
    yield 'moebius (twists=2)', o3d.geometry.TriangleMesh.create_moebius(
        twists=2)
    yield 'moebius (twists=3)', o3d.geometry.TriangleMesh.create_moebius(
        twists=3)

    yield 'knot', meshes.knot()

    if edge_cases:
        yield 'non-manifold edge', meshes.non_manifold_edge()
        yield 'non-manifold vertex', meshes.non_manifold_vertex()
        yield 'open box', meshes.open_box()
        yield 'boxes', meshes.intersecting_boxes()


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
    watertight = mesh.is_watertight()
    print('  watertight:             %s' % fmt_bool(watertight))
    orientable = mesh.is_orientable()
    print('  orientable:             %s' % fmt_bool(orientable))

    mesh.compute_vertex_normals()

    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        print('  # visualize non-manifold edges (allow_boundary_edges=True)')
        o3d.visualization.draw_geometries(
            [mesh, meshes.edges_to_lineset(mesh, edges, (1, 0, 0))])
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        print('  # visualize non-manifold edges (allow_boundary_edges=False)')
        o3d.visualization.draw_geometries(
            [mesh, meshes.edges_to_lineset(mesh, edges, (0, 1, 0))])
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
            [mesh, meshes.edges_to_lineset(mesh, edges, (1, 1, 0))])
    if watertight:
        print('  # visualize watertight mesh')
        o3d.visualization.draw_geometries([mesh])

    if not edge_manifold:
        print('  # Remove non-manifold edges')
        mesh.remove_non_manifold_edges()
        print('  # Is mesh now edge-manifold: {}'.format(
            fmt_bool(mesh.is_edge_manifold())))
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    # test mesh properties
    print('#' * 80)
    print('Test mesh properties')
    print('#' * 80)
    for name, mesh in mesh_generator(edge_cases=True):
        check_properties(name, mesh)

    # fix triangle orientation
    print('#' * 80)
    print('Fix triangle orientation')
    print('#' * 80)
    for name, mesh in mesh_generator(edge_cases=False):
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
    bbox = o3d.geometry.TriangleMesh.create_box(20, 20, 20).translate(
        (-10, -10, -10))
    meshes = [o3d.geometry.TriangleMesh.create_box() for _ in range(20)]
    meshes.append(o3d.geometry.TriangleMesh.create_sphere())
    meshes.append(o3d.geometry.TriangleMesh.create_cone())
    meshes.append(o3d.geometry.TriangleMesh.create_torus())
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
