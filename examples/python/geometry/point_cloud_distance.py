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
    sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.pointcloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(
        sample_ply_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)

    chair.paint_uniform_color([0, 0, 1])
    pcd.paint_uniform_color([1, 0, 0])
    print("Displaying the two point clouds used for calculating distance ...")
    o3d.visualization.draw([pcd, chair])

    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    print("Printing average distance between the two point clouds ...")
    print(dists)
