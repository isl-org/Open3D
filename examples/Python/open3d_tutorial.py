# Helpers and monkey patches for ipynb tutorials

import open3d as o3d
import numpy as np
import PIL.Image
import IPython.display
import os
import urllib
import tarfile
import gzip
import zipfile
import shutil

interactive = True


def jupyter_draw_geometries(
        geoms,
        window_name="Open3D",
        width=1920,
        height=1080,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False,
        lookat=None,
        up=None,
        front=None,
        zoom=None,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        visible=True,  # If false, capture_screen_float_buffer() won't work.
    )
    vis.get_render_option().point_show_normal = point_show_normal
    vis.get_render_option().mesh_show_wireframe = mesh_show_wireframe
    vis.get_render_option().mesh_show_back_face = mesh_show_back_face
    for geom in geoms:
        vis.add_geometry(geom)
    if lookat is not None:
        vis.get_view_control().set_lookat(lookat)
    if up is not None:
        vis.get_view_control().set_up(up)
    if front is not None:
        vis.get_view_control().set_front(front)
    if zoom is not None:
        vis.get_view_control().set_zoom(zoom)
    if interactive:
        vis.run()
    else:
        for geom in geoms:
            vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
    im = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im = (255 * np.asarray(im)).astype(np.uint8)
    IPython.display.display(PIL.Image.fromarray(im, "RGB"))


o3d.visualization.draw_geometries = jupyter_draw_geometries


def edges_to_lineset(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _relative_path(path):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, path)


def download_fountain_dataset():
    fountain_path = _relative_path("../TestData/fountain_small")
    fountain_zip_path = _relative_path("../TestData/fountain.zip")
    if not os.path.exists(fountain_path):
        print("downloading fountain dataset")
        url = "https://storage.googleapis.com/isl-datasets/open3d-dev/fountain.zip"
        urllib.request.urlretrieve(url, fountain_zip_path)
        print("extract fountain dataset")
        with zipfile.ZipFile(fountain_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(fountain_path))
        os.remove(fountain_zip_path)
    return fountain_path


def get_non_manifold_edge_mesh():
    verts = np.array(
        [[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)))
    return mesh


def get_non_manifold_vertex_mesh():
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
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)))
    return mesh


def get_open_box_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:-2])
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((0.8 * np.pi, 0, 0.66 * np.pi)))
    return mesh


def get_intersecting_boxes_mesh():
    mesh0 = o3d.geometry.TriangleMesh.create_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = o3d.geometry.TriangleMesh.create_box()
    mesh1.transform(T)
    mesh = mesh0 + mesh1
    mesh.compute_vertex_normals()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz(
        (0.7 * np.pi, 0, 0.6 * np.pi)))
    return mesh


def get_armadillo_mesh():
    armadillo_path = _relative_path("../TestData/Armadillo.ply")
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


def get_bunny_mesh():
    bunny_path = _relative_path("../TestData/Bunny.ply")
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


def get_knot_mesh():
    mesh = o3d.io.read_triangle_mesh(_relative_path("../TestData/knot.ply"))
    mesh.compute_vertex_normals()
    return mesh


def get_eagle_pcd():
    path = _relative_path("../TestData/eagle.ply")
    if not os.path.exists(path):
        print("downloading eagle pcl")
        url = "http://www.cs.jhu.edu/~misha/Code/PoissonRecon/eagle.points.ply"
        urllib.request.urlretrieve(url, path)
    pcd = o3d.io.read_point_cloud(path)
    return pcd
