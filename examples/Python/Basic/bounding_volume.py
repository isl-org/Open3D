# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/bounding_volume.py

import numpy as np
import open3d as o3d

import meshes

np.random.seed(42)


def mesh_generator():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.rotate((0.3, 0.5, 0.1))
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
