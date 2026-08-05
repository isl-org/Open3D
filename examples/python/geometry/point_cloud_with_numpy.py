# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    # Generate some n x 3 matrix using a variant of sync function.
    x = np.linspace(-3, 3, 201)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print("Printing numpy array used to make Open3D pointcloud ...")
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])

    # Convert Open3D.o3d.geometry.PointCloud to numpy array.
    xyz_converted = np.asarray(pcd.points)
    print("Printing numpy array made using Open3D pointcloud ...")
    print(xyz_converted)
