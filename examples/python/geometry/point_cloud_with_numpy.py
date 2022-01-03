# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
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
