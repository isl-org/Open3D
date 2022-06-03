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
"""Doppler ICP (Iterative Closest Point) registration algorithm"""

import argparse
import copy

import open3d as o3d
import numpy as np


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str,
                        help='Path to source point cloud in XYZD format')
    parser.add_argument('target', type=str,
                        help='Path to target point cloud in XYZD format')

    args = parser.parse_args()

    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)

    # Compute direction vectors for the source point cloud.
    points = np.asarray(source.points)
    directions = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    source_directions = o3d.utility.Vector3dVector(directions)

    # Compute normal vectors for the target point cloud.
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))

    # This is the implementation of the following paper:
    # B. Hexsel, H. Vhavle, Y. Chen,
    # DICP: Doppler Iterative Closest Point Algorithm, RSS 2022.
    print('Apply point-to-plane Doppler ICP')
    max_corr_distance = 0.3  # meters
    init_transform = np.eye(4)
    result = o3d.pipelines.registration.registration_doppler_icp(
        source, target, source_directions, max_corr_distance, init_transform,
        o3d.pipelines.registration.TransformationEstimationForDopplerICP(
            lambda_doppler=0.01,
            reject_dynamic_outliers=False,
            doppler_outlier_threshold=2.0,  # m/s
            outlier_rejection_min_iteration=2,
            geometric_robust_loss_min_iteration=0,
            doppler_robust_loss_min_iteration=2,
            geometric_kernel=o3d.pipelines.registration.TukeyLoss(k=0.5),
            doppler_kernel=o3d.pipelines.registration.TukeyLoss(k=0.5)),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50),
        period=0.1,  # seconds
        T_V_to_S=np.eye(4),  # vehicle-to-sensor extrinsic calibration
        )
    print(result)
    print('Transformation is:')
    print(result.transformation, "\n")
    draw_registration_result(source, target, result.transformation)
