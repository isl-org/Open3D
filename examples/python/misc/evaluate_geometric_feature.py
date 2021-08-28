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

# examples/python/misc/evaluate_geometric_feature.py

import open3d as o3d
import numpy as np


def evaluate(pcd_target, pcd_source, feature_target, feature_source):
    tree_target = o3d.geometry.KDTreeFlann(feature_target)
    pt_dis = np.zeros(len(pcd_source.points))
    for i in range(len(pcd_source.points)):
        [_, idx,
         _] = tree_target.search_knn_vector_xd(feature_source.data[:, i], 1)
        pt_dis[i] = np.linalg.norm(pcd_source.points[i] -
                                   pcd_target.points[idx[0]])
    return pt_dis


if __name__ == "__main__":
    pcd_target = o3d.io.read_point_cloud(
        "../../test_data/Feature/cloud_bin_0.pcd")
    pcd_source = o3d.io.read_point_cloud(
        "../../test_data/Feature/cloud_bin_1.pcd")
    feature_target = o3d.io.read_feature(
        "../../test_data/Feature/cloud_bin_0.fpfh.bin")
    feature_source = o3d.io.read_feature(
        "../../test_data/Feature/cloud_bin_1.fpfh.bin")
    pt_dis = evaluate(pcd_target, pcd_source, feature_target, feature_source)
    num_good = sum(pt_dis < 0.075)
    print(
        "{:.2f}% points in source pointcloud successfully found their correspondence."
        .format(num_good * 100.0 / len(pcd_source.points)))
