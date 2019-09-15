# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/meshes.py

import numpy as np
import open3d as o3d
import os
import urllib.request
import gzip
import tarfile
import shutil
import time


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def apply_noise(mesh, noise):
    vertices = np.asarray(mesh.vertices)
    vertices += np.random.uniform(-noise, noise, size=vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def triangle():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([(np.sqrt(8 / 9), 0, -1 / 3),
                  (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                  (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3)],
                 dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    mesh.compute_vertex_normals()
    return mesh


def plane():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [0, 0.2, 0], [1, 0.2, 0], [1, 0, 0]],
                 dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [2, 0,
                                                                      3]]))
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_edge():
    verts = np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
                     dtype=np.float64)
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_vertex():
    verts = np.array([[-1, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 0],
                      [-1, 0, 1], [1, 0, 1], [0, 1, 1]],
                     dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3], [4, 5, 6],
                          [4, 5, 3], [5, 6, 3], [4, 6, 3]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def open_box():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:-2])
    mesh.compute_vertex_normals()
    return mesh


def intersecting_boxes():
    mesh0 = o3d.geometry.TriangleMesh.create_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = o3d.geometry.TriangleMesh.create_box()
    mesh1.transform(T)
    mesh = mesh0 + mesh1
    mesh.compute_vertex_normals()
    return mesh


def _relative_path(path):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, path)


def knot():
    mesh = o3d.io.read_triangle_mesh(_relative_path('../../TestData/knot.ply'))
    mesh.compute_vertex_normals()
    return mesh


def bathtub():
    mesh = o3d.io.read_triangle_mesh(
        _relative_path('../../TestData/bathtub_0154.ply'))
    mesh.compute_vertex_normals()
    return mesh


def armadillo():
    armadillo_path = _relative_path('../../TestData/Armadillo.ply')
    if not os.path.exists(armadillo_path):
        print('downloading armadillo mesh')
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz'
        urllib.request.urlretrieve(url, armadillo_path + '.gz')
        print('extract armadillo mesh')
        with gzip.open(armadillo_path + '.gz', 'rb') as fin:
            with open(armadillo_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + '.gz')
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh


def bunny():
    bunny_path = _relative_path('../../TestData/Bunny.ply')
    if not os.path.exists(bunny_path):
        print('downloading bunny mesh')
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        urllib.request.urlretrieve(url, bunny_path + '.tar.gz')
        print('extract bunny mesh')
        with tarfile.open(bunny_path + '.tar.gz') as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(os.path.dirname(bunny_path), 'bunny', 'reconstruction',
                         'bun_zipper.ply'), bunny_path)
        os.remove(bunny_path + '.tar.gz')
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), 'bunny'))
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    mesh.compute_vertex_normals()
    return mesh


def center_and_scale(mesh):
    vertices = np.asarray(mesh.vertices)
    vertices = vertices / max(vertices.max(axis=0) - vertices.min(axis=0))
    vertices -= vertices.mean(axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


if __name__ == '__main__':

    def process(mesh):
        mesh.compute_vertex_normals()
        mesh = center_and_scale(mesh)
        return mesh

    print('visualize')
    print('  tetrahedron, octahedron, icosahedron')
    print('  torus, moebius strip one twist, moebius strip two twists')
    d = 1.5
    geoms = [
        process(o3d.geometry.TriangleMesh.create_tetrahedron()).translate(
            (-d, 0, 0)),
        process(o3d.geometry.TriangleMesh.create_octahedron()).translate(
            (0, 0, 0)),
        process(o3d.geometry.TriangleMesh.create_icosahedron()).translate(
            (d, 0, 0)),
        process(o3d.geometry.TriangleMesh.create_torus()).translate(
            (-d, -d, 0)),
        process(o3d.geometry.TriangleMesh.create_moebius(twists=1)).translate(
            (0, -d, 0)),
        process(o3d.geometry.TriangleMesh.create_moebius(twists=2)).translate(
            (d, -d, 0)),
    ]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True
    for geom in geoms:
        vis.add_geometry(geom)

    scales = [0.995 for _ in range(100)] + [1 / 0.995 for _ in range(100)]
    axisangles = [(0.2 / np.sqrt(2), 0.2 / np.sqrt(2), 0) for _ in range(200)]

    for scale, aa in zip(scales, axisangles):
        for geom in geoms:
            geom.scale(scale).rotate(aa,
                                     center=True,
                                     type=o3d.geometry.RotationType.AxisAngle)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
