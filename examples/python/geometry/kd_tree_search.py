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
"""Build a KDTree and use it for neighbour search"""

import open3d as o3d
import numpy as np


def radius_search():
    print("Loading pointcloud ...")
    sample_pcd_data = o3d.data.SamplePointCloudPCD()
    pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the neighbors of 50000th point with distance less than 0.2, and painting them green ..."
    )
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50000], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


def knn_search():
    print("Loading pointcloud ...")
    sample_pcd = o3d.data.SamplePointCloudPCD()
    pcd = o3d.io.read_point_cloud(sample_pcd.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the 2000 nearest neighbors of 50000th point, and painting them red ..."
    )
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[50000], 2000)
    np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


if __name__ == "__main__":
    knn_search()
    radius_search()
