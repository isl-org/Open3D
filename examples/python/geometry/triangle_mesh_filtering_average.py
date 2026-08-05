# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from numpy.random.mtrand import laplace
import open3d as o3d
import numpy as np


def average_filtering():
    # Create noisy mesh.
    knot_mesh = o3d.data.KnotMesh()
    mesh_in = o3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices)
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of average mesh filter after 1 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of average mesh filter after 5 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])


def laplace_filtering():
    # Create noisy mesh.
    knot_mesh = o3d.data.KnotMesh()
    mesh_in = o3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices)
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Laplace mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Laplace mesh filter after 50 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])


def taubin_filtering():
    # Create noisy mesh.
    knot_mesh = o3d.data.KnotMesh()
    mesh_in = o3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices)
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    o3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Taubin mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Taubin mesh filter after 100 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
    mesh_out.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_out])


if __name__ == "__main__":
    average_filtering()
    laplace_filtering()
    taubin_filtering()
