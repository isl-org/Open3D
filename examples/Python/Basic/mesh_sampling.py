# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
import time
import open3d as o3d

import meshes


def time_fcn(fcn, *fcn_args, runs=5):
    times = []
    for _ in range(runs):
        tic = time.time()
        res = fcn(*fcn_args)
        times.append(time.time() - tic)
    return res, times


def mesh_generator():
    yield meshes.plane()
    yield o3d.geometry.TriangleMesh.create_sphere()
    yield meshes.bunny()
    yield meshes.armadillo()


if __name__ == "__main__":
    plane = meshes.plane()
    o3d.visualization.draw_geometries([plane])

    print('Uniform sampling can yield clusters of points on the surface')
    pcd = plane.sample_points_uniformly(number_of_points=500)
    o3d.visualization.draw_geometries([pcd])

    print(
        'Poisson disk sampling can evenly distributes the points on the surface.'
    )
    print('The method implements sample elimination.')
    print('Therefore, the method starts with a sampled point cloud and removes '
          'point to satisfy the sampling criterion.')
    print('The method supports two options to provide the initial point cloud')
    print('1) Default via the parameter init_factor: The method first samples '
          'uniformly a point cloud from the mesh with '
          'init_factor x number_of_points and uses this for the elimination')
    pcd = plane.sample_points_poisson_disk(number_of_points=500, init_factor=5)
    o3d.visualization.draw_geometries([pcd])

    print(
        '2) one can provide an own point cloud and pass it to the '
        'o3d.geometry.sample_points_poisson_disk method. Then this point cloud is used '
        'for elimination.')
    print('Initial point cloud')
    pcd = plane.sample_points_uniformly(number_of_points=2500)
    o3d.visualization.draw_geometries([pcd])
    pcd = plane.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
    o3d.visualization.draw_geometries([pcd])

    print('Timings')
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

        pcd, times = time_fcn(mesh.sample_points_uniformly, 500)
        print('sample uniform took on average: %f[s]' % np.mean(times))
        o3d.visualization.draw_geometries([pcd])

        pcd, times = time_fcn(mesh.sample_points_poisson_disk, 500, 5)
        print('sample poisson disk took on average: %f[s]' % np.mean(times))
        o3d.visualization.draw_geometries([pcd])
