# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

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
        "../../TestData/Feature/cloud_bin_0.pcd")
    pcd_source = o3d.io.read_point_cloud(
        "../../TestData/Feature/cloud_bin_1.pcd")
    feature_target = o3d.io.read_feature(
        "../../TestData/Feature/cloud_bin_0.fpfh.bin")
    feature_source = o3d.io.read_feature(
        "../../TestData/Feature/cloud_bin_1.fpfh.bin")
    pt_dis = evaluate(pcd_target, pcd_source, feature_target, feature_source)
    num_good = sum(pt_dis < 0.075)
    print(
        "{:.2f}% points in source pointcloud successfully found their correspondence."
        .format(num_good * 100.0 / len(pcd_source.points)))
