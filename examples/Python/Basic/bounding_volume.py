# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/bounding_volume.py

import numpy as np
import open3d as o3d
import os

import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../Misc"))
import meshes

np.random.seed(42)


def mesh_generator():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0.3, 0.5, 0.1)))
    yield "rotated box mesh", mesh
    yield "rotated box pcd", mesh.sample_points_uniformly(500)

    mesh = meshes.armadillo()
    yield "armadillo mesh", mesh
    yield "armadillo pcd", mesh.sample_points_uniformly(500)


if __name__ == "__main__":
    for name, geom in mesh_generator():
        aabox = geom.get_axis_aligned_bounding_box()
        print("%s has an axis aligned box volume of %f" %
              (name, aabox.volume()))
        obox = geom.get_oriented_bounding_box()
        print("%s has an oriented box volume of %f" % (name, obox.volume()))
        aabox.color = [1, 0, 0]
        obox.color = [0, 1, 0]
        o3d.visualization.draw_geometries([geom, aabox, obox])

    mesh = meshes.armadillo()

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-30, 0, -10),
                                               max_bound=(10, 20, 10))
    o3d.visualization.draw_geometries([mesh, bbox])
    o3d.visualization.draw_geometries([mesh.crop(bbox), bbox])

    bbox = o3d.geometry.OrientedBoundingBox(
        center=(-10, 10, 0),
        R=bbox.get_rotation_matrix_from_xyz((2, 1, 0)),
        extent=(40, 20, 20),
    )
    o3d.visualization.draw_geometries([mesh, bbox])
    o3d.visualization.draw_geometries([mesh.crop(bbox), bbox])

    pcd = mesh.sample_points_uniformly(500000)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-30, 0, -10),
                                               max_bound=(10, 20, 10))
    o3d.visualization.draw_geometries([pcd, bbox])
    o3d.visualization.draw_geometries([pcd.crop(bbox), bbox])

    bbox = o3d.geometry.OrientedBoundingBox(
        center=(-10, 10, 0),
        R=bbox.get_rotation_matrix_from_xyz((2, 1, 0)),
        extent=(40, 20, 20),
    )
    o3d.visualization.draw_geometries([pcd, bbox])
    o3d.visualization.draw_geometries([pcd.crop(bbox), bbox])
