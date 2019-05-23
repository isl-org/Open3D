# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
import os
import urllib.request
import gzip
import tarfile
import shutil
import time
import open3d as o3d

def create_mesh_plane():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
            np.array([[0,0,0], [0,0.2,0], [1,0.2,0], [1,0,0]], dtype=np.float32))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,2,1], [2,0,3]]))
    return mesh

def armadillo_mesh():
    armadillo_path = '../../TestData/Armadillo.ply'
    if not os.path.exists(armadillo_path):
        print('downloading armadillo mesh')
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz'
        urllib.request.urlretrieve(url, armadillo_path + '.gz')
        print('extract armadillo mesh')
        with gzip.open(armadillo_path + '.gz', 'rb') as fin:
            with open(armadillo_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + '.gz')
    return o3d.io.read_triangle_mesh(armadillo_path)

def bunny_mesh():
    bunny_path = '../../TestData/Bunny.ply'
    if not os.path.exists(bunny_path):
        print('downloading bunny mesh')
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        urllib.request.urlretrieve(url, bunny_path + '.tar.gz')
        print('extract bunny mesh')
        with tarfile.open(bunny_path + '.tar.gz') as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
                os.path.join(os.path.dirname(bunny_path),
                    'bunny', 'reconstruction', 'bun_zipper.ply'),
                bunny_path)
        os.remove(bunny_path + '.tar.gz')
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), 'bunny'))
    return o3d.io.read_triangle_mesh(bunny_path)

def mesh_generator():
    yield o3d.geometry.create_mesh_box()
    yield o3d.geometry.create_mesh_sphere()
    yield o3d.io.read_triangle_mesh('../../TestData/knot.ply')
    yield bunny_mesh()
    yield armadillo_mesh()

if __name__ == "__main__":
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        hull = o3d.geometry.compute_mesh_convex_hull(mesh)
        hull_ls = o3d.geometry.create_line_set_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1,0,0))
        o3d.visualization.draw_geometries([mesh, hull_ls])

        pcl = o3d.geometry.sample_points_poisson_disk(mesh, number_of_points=2000)
        hull = o3d.geometry.compute_point_cloud_convex_hull(pcl)
        hull_ls = o3d.geometry.create_line_set_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1,0,0))
        o3d.visualization.draw_geometries([pcl, hull_ls])

