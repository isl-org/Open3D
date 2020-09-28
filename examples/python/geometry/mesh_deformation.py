# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/geometry/mesh_deformation.py

import numpy as np
import open3d as o3d
import time
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes


def problem0():
    mesh = meshes.plane(height=1, width=1)
    mesh = mesh.subdivide_midpoint(3)
    vertices = np.asarray(mesh.vertices)
    static_ids = [
        1, 46, 47, 48, 16, 51, 49, 50, 6, 31, 33, 32, 11, 26, 27, 25, 0, 64, 65,
        20, 66, 68, 67, 7, 69, 71, 70, 22, 72, 74, 73, 3, 15, 44, 43, 45, 5, 41,
        40, 42, 13, 39, 37, 38, 2, 56, 55, 19, 61, 60, 59, 8, 76, 75, 77, 23
    ]
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [4]
    handle_positions = [vertices[4] + np.array((0, 0, 0.4))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


def problem1():
    mesh = meshes.plane(height=1, width=1)
    mesh = mesh.subdivide_midpoint(3)
    vertices = np.asarray(mesh.vertices)
    static_ids = [
        1, 46, 15, 43, 5, 40, 13, 38, 2, 56, 37, 39, 42, 41, 45, 44, 48, 47
    ]
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [21]
    handle_positions = [vertices[21] + np.array((0, 0, 0.4))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


def problem2():
    mesh = meshes.armadillo()
    vertices = np.asarray(mesh.vertices)
    static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [2490]
    handle_positions = [vertices[2490] + np.array((-40, -40, -40))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.Debug)

    for mesh, constraint_ids, constraint_pos in [
            problem0(), problem1(), problem2()
    ]:
        constraint_ids = np.array(constraint_ids, dtype=np.int32)
        constraint_pos = o3d.utility.Vector3dVector(constraint_pos)
        tic = time.time()
        mesh_prime = mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(constraint_ids), constraint_pos, max_iter=50)
        print("deform took {}[s]".format(time.time() - tic))
        mesh_prime.compute_vertex_normals()

        mesh.paint_uniform_color((1, 0, 0))
        handles = o3d.geometry.PointCloud()
        handles.points = constraint_pos
        handles.paint_uniform_color((0, 1, 0))
        o3d.visualization.draw_geometries([mesh, mesh_prime, handles])
