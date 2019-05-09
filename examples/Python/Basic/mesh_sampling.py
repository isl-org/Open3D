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
from open3d import *

def create_mesh_plane():
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(
            np.array([[0,0,0], [0,0.2,0], [1,0.2,0], [1,0,0]], dtype=np.float32))
    mesh.triangles = Vector3iVector(np.array([[0,2,1], [2,0,3]]))
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
    return read_triangle_mesh(armadillo_path)

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
    return read_triangle_mesh(bunny_path)

def time_fcn(fcn, *fcn_args, runs=3):
    times = []
    for _ in range(runs):
        tic = time.time()
        res = fcn(*fcn_args)
        times.append(time.time() - tic)
    return res, times

def mesh_generator():
    yield create_mesh_plane()
    yield create_mesh_sphere()
    yield bunny_mesh()
    yield armadillo_mesh()

if __name__ == "__main__":
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        draw_geometries([mesh])

        pcd, times = time_fcn(sample_points_uniformly, mesh, 500)
        print('sample uniform took on average: %f[s]' % np.mean(times))
        draw_geometries([pcd])

        pcd, times = time_fcn(sample_points_poisson_disk, mesh, 500, 5)
        print('sample poisson disk took on average: %f[s]' % np.mean(times))
        draw_geometries([pcd])
