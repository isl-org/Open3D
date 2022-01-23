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

import os
import sys
import numpy as np
import time

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import *

do_visualization = False


def get_ply_path(dataset_name, id):
    return "%s/%s/cloud_bin_%d.ply" % (dataset_path, dataset_name, id)


def get_log_path(dataset_name):
    return "%s/ransac_%s.log" % (dataset_path, dataset_name)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("\n:: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    start = time.time()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print(result)
    print("RANSAC global registration took %.3f sec." % (time.time() - start))
    draw_registration_result(source_down, target_down, result.transformation)


dataset_path = os.path.join(os.path.dirname(pyexample_path), 'test_data',
                            'benchmark_data')
dataset_names = ['livingroom1', 'livingroom2', 'office1', 'office2']

if __name__ == "__main__":
    # data preparation
    get_redwood_dataset()
    voxel_size = 0.05

    # do RANSAC based alignment
    for dataset_name in dataset_names:
        ply_file_names = get_file_list("%s/%s/" % (dataset_path, dataset_name),
                                       ".ply")
        n_ply_files = len(ply_file_names)

        alignment = []
        for s in range(n_ply_files):
            for t in range(s + 1, n_ply_files):

                print("%s:: matching %d-%d" % (dataset_name, s, t))
                source = o3d.io.read_point_cloud(get_ply_path(dataset_name, s))
                target = o3d.io.read_point_cloud(get_ply_path(dataset_name, t))
                source_down, source_fpfh = preprocess_point_cloud(
                    source, voxel_size)
                target_down, target_fpfh = preprocess_point_cloud(
                    target, voxel_size)

                result = execute_global_registration(source_down, target_down,
                                                     source_fpfh, target_fpfh,
                                                     voxel_size)
                if (result.transformation.trace() == 4.0):
                    success = False
                else:
                    success = True

                # Note: we save inverse of result.transformation
                # to comply with http://redwood-data.org/indoor/fileformat.html
                if not success:
                    print("No reasonable solution.")
                else:
                    alignment.append(
                        CameraPose([s, t, n_ply_files],
                                   np.linalg.inv(result.transformation)))
                    print(np.linalg.inv(result.transformation))

                if do_visualization:
                    draw_registration_result(source_down, target_down,
                                             result.transformation)
        write_trajectory(alignment, get_log_path(dataset_name))

    # do evaluation
