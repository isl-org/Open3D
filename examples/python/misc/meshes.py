# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/misc/meshes.py

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
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.array(
                [
                    (np.sqrt(8 / 9), 0, -1 / 3),
                    (-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3),
                    (-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3),
                ],
                dtype=np.float32,
            )),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 1, 2]])),
    )
    mesh.compute_vertex_normals()
    return mesh


def plane(height=0.2, width=1):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(
            np.array(
                [[0, 0, 0], [0, height, 0], [width, height, 0], [width, 0, 0]],
                dtype=np.float32,
            )),
        triangles=o3d.utility.Vector3iVector(np.array([[0, 2, 1], [2, 0, 3]])),
    )
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_edge():
    verts = np.array(
        [[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def non_manifold_vertex():
    verts = np.array(
        [
            [-1, 0, -1],
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, 0],
            [-1, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    triangles = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
        [4, 5, 6],
        [4, 5, 3],
        [5, 6, 3],
        [4, 6, 3],
    ])
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
    mesh = o3d.io.read_triangle_mesh(_relative_path("../../test_data/knot.ply"))
    mesh.compute_vertex_normals()
    return mesh


def bathtub():
    mesh = o3d.io.read_triangle_mesh(
        _relative_path("../../test_data/bathtub_0154.ply"))
    mesh.compute_vertex_normals()
    return mesh


def armadillo():
    armadillo_path = _relative_path("../../test_data/Armadillo.ply")
    if not os.path.exists(armadillo_path):
        print("downloading armadillo mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
        urllib.request.urlretrieve(url, armadillo_path + ".gz")
        print("extract armadillo mesh")
        with gzip.open(armadillo_path + ".gz", "rb") as fin:
            with open(armadillo_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + ".gz")
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh


def bunny():
    bunny_path = _relative_path("../../test_data/Bunny.ply")
    if not os.path.exists(bunny_path):
        print("downloading bunny mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        urllib.request.urlretrieve(url, bunny_path + ".tar.gz")
        print("extract bunny mesh")
        with tarfile.open(bunny_path + ".tar.gz") as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(
                os.path.dirname(bunny_path),
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            bunny_path,
        )
        os.remove(bunny_path + ".tar.gz")
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), "bunny"))
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    mesh.compute_vertex_normals()
    return mesh


def eagle():
    path = _relative_path("../../test_data/eagle.ply")
    if not os.path.exists(path):
        print("downloading eagle pcl")
        url = "http://www.cs.jhu.edu/~misha/Code/PoissonRecon/eagle.points.ply"
        urllib.request.urlretrieve(url, path)
    pcd = o3d.io.read_point_cloud(path)
    return pcd


def center_and_scale(mesh):
    vertices = np.asarray(mesh.vertices)
    vertices = vertices / max(vertices.max(axis=0) - vertices.min(axis=0))
    vertices -= vertices.mean(axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def print_1D_array_for_cpp(prefix, array):
    if array.dtype == np.float32:
        dtype = "float"
    elif array.dtype == np.float64:
        dtype = "double"
    elif array.dtype == np.int32:
        dtype = "int"
    elif array.dtype == np.uint32:
        dtype = "size_t"
    elif array.dtype == np.bool:
        dtype = "bool"
    else:
        raise Exception("invalid dtype")
    print("std::vector<{}> {} = {{".format(dtype, prefix))
    print(", ".join(map(str, array)))
    print("};")


def print_2D_array_for_cpp(prefix, values, fmt):
    if values.shape[0] > 0:
        print("{} = {{".format(prefix))
        # e.g. if fmt == ".6f", v3d_fmt == "  {{{0:.6f}, {0:.6f}, {0:.6f}}}'
        v3d_fmt = "  {{{0:%s}, {0:%s}, {0:%s}}}" % (".6f", ".6f", ".6f")
        print(",\n".join([v3d_fmt.format(v[0], v[1], v[2]) for v in values]))
        print("};")


def print_mesh_for_cpp(mesh, prefix=""):
    print_2D_array_for_cpp("{}vertices_".format(prefix),
                           np.asarray(mesh.vertices), ".6f")
    print_2D_array_for_cpp("{}vertex_normals_".format(prefix),
                           np.asarray(mesh.vertex_normals), ".6f")
    print_2D_array_for_cpp("{}vertex_colors_".format(prefix),
                           np.asarray(mesh.vertex_colors), ".6f")
    print_2D_array_for_cpp("{}triangles_".format(prefix),
                           np.asarray(mesh.triangles), "d")
    print_2D_array_for_cpp("{}triangle_normals_".format(prefix),
                           np.asarray(mesh.triangle_normals), ".6f")
