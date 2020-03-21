# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/mesh_voxelization.py

import os
import sys
import urllib.request
import gzip
import tarfile
import shutil
import time
import open3d as o3d
import numpy as np
import numpy.matlib

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes


def preprocess(model):
    """Normalize model to fit in unit sphere (sphere with unit radius).
    
    Calculate center & scale of vertices, and transform vertices to have 0 mean and unit variance. 

    Returns:
        open3d.geometry.TriangleMesh: normalized mesh
    """
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= np.matlib.repmat(center, len(model.vertices), 1)
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)

    ## Paint uniform color for pleasing visualization
    model.paint_uniform_color(np.array([1, 0.7, 0]))
    return model


def mesh_generator():
    """Yields meshes
    Yields:
        open3d.geometry.TriangleMesh: yielded mesh 
    """
    yield meshes.armadillo()
    yield meshes.bunny()


if __name__ == "__main__":
    print("Start mesh_sampling_and_voxelization example")
    for mesh in mesh_generator():
        print("Normalize mesh")
        mesh = preprocess(mesh)
        o3d.visualization.draw_geometries([mesh])
        print("")

        print("Sample uniform points")
        start = time.time()
        pcd = mesh.sample_points_uniformly(number_of_points=100000)
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        print("")

        print("visualize sampled point cloud")
        o3d.visualization.draw_geometries([pcd])
        print("")

        print("Voxelize point cloud")
        start = time.time()
        voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                               voxel_size=0.05)
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        print("")

        print("visualize voxel grid")
        o3d.visualization.draw_geometries([voxel])
        print("")

        print("Save and load voxel grid")
        print(voxel)
        start = time.time()
        o3d.io.write_voxel_grid("save.ply", voxel)
        voxel_load = o3d.io.read_voxel_grid("save.ply")
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        print(voxel_load)
        print(voxel_load.voxel_size)
        print(voxel_load.origin)
        print("")

        print("Element-wise check if points belong to voxel grid")
        queries = np.asarray(pcd.points)
        start = time.time()
        output = voxel_load.check_if_included(
            o3d.utility.Vector3dVector(queries))
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        print(output[:10])
        print("")

        print(
            "Element-wise check if points with additive Gaussian noise belong to voxel grid"
        )
        queries_noise = queries + np.random.normal(0, 0.1, (len(pcd.points), 3))
        start = time.time()
        output_noise = voxel_load.check_if_included(
            o3d.utility.Vector3dVector(queries_noise))
        print(output_noise[:10])
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        print("")

        print("Transform voxelgrid to octree")
        start = time.time()
        octree = voxel_load.to_octree(max_depth=8)
        print(octree)
        print("took %.2f milliseconds" % ((time.time() - start) * 1000.0))
        o3d.visualization.draw_geometries([octree])
        print("")
