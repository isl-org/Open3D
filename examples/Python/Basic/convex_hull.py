# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/convex_hull.py

import numpy as np
import os
import urllib.request
import gzip
import tarfile
import shutil
import time
import open3d as o3d

import meshes


def mesh_generator():
    yield o3d.geometry.create_mesh_box()
    yield o3d.geometry.create_mesh_sphere()
    yield meshes.knot()
    yield meshes.bunny()
    yield meshes.armadillo()


if __name__ == "__main__":
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        hull = o3d.geometry.compute_mesh_convex_hull(mesh)
        hull_ls = o3d.geometry.create_line_set_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([mesh, hull_ls])

        pcl = o3d.geometry.sample_points_poisson_disk(mesh,
                                                      number_of_points=2000)
        hull = o3d.geometry.compute_point_cloud_convex_hull(pcl)
        hull_ls = o3d.geometry.create_line_set_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        o3d.visualization.draw_geometries([pcl, hull_ls])
