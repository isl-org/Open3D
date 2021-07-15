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
import open3d.core as o3c
import numpy as np
import pytest

import sys
import os

test_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"
sys.path.append(test_path)
test_data_path = test_path + "../../examples/test_data/"

source_pointcloud_filename = test_data_path + "ICP/cloud_bin_0.pcd"
target_pointcloud_filename = test_data_path + "ICP/cloud_bin_1.pcd"

from open3d_test import list_devices


def get_pcds(dtype, device):
    source = o3d.t.io.read_point_cloud(source_pointcloud_filename)
    target = o3d.t.io.read_point_cloud(target_pointcloud_filename)

    source = source.to(device)
    target = target.to(device)

    source.point["points"] = source.point["points"].to(dtype)
    source.point["normals"] = source.point["normals"].to(dtype)
    target.point["points"] = target.point["points"].to(dtype)
    target.point["normals"] = target.point["normals"].to(dtype)

    return source, target


@pytest.mark.parametrize("device", list_devices())
def test_icp_convergence_criteria_constructor(device):

    # Constructor.
    convergence_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria()

    # Checking default values.
    assert convergence_criteria.max_iteration == 30
    assert convergence_criteria.relative_fitness == 1e-06
    assert convergence_criteria.relative_rmse == 1e-06


@pytest.mark.parametrize("device", list_devices())
def test_registration_result_constructor(device):
    dtype = o3c.Dtype.Float64

    # Constructor.
    registration_result = o3d.t.pipelines.registration.RegistrationResult()

    # Checking default values.
    assert registration_result.inlier_rmse == 0.0
    assert registration_result.fitness == 0.0
    assert registration_result.transformation.allclose(
        o3c.Tensor.eye(4, dtype, o3c.Device("CPU:0")))


@pytest.mark.parametrize("device", list_devices())
def test_evaluate_registration(device):

    supported_dtypes = [o3c.Dtype.Float32, o3c.Dtype.Float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy_pointcloud()
        target_legacy = target_t.to_legacy_pointcloud()

        max_correspondence_distance = 1.25
        init_trans_legacy = np.eye(4)
        init_trans_t = o3c.Tensor.eye(4, o3c.Dtype.Float64, device)

        evaluation_t = o3d.t.pipelines.registration.evaluate_registration(
            source_t, target_t, max_correspondence_distance, init_trans_t)
        evaluation_legacy = o3d.pipelines.registration.evaluate_registration(
            source_legacy, target_legacy, max_correspondence_distance,
            init_trans_legacy)

        np.testing.assert_allclose(evaluation_t.inlier_rmse,
                                   evaluation_legacy.inlier_rmse, 0.001)
        np.testing.assert_allclose(evaluation_t.fitness,
                                   evaluation_legacy.fitness, 0.001)


@pytest.mark.parametrize("device", list_devices())
def test_registration_icp_point_to_point(device):

    supported_dtypes = [o3c.Dtype.Float32, o3c.Dtype.Float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy_pointcloud()
        target_legacy = target_t.to_legacy_pointcloud()

        max_correspondence_distance = 1.5

        init_trans_legacy = np.array([[0.862, 0.011, -0.507, 0.5],
                                      [-0.139, 0.967, -0.215, 0.7],
                                      [0.487, 0.255, 0.835, -1.4],
                                      [0.0, 0.0, 0.0, 1.0]])
        init_trans_t = o3c.Tensor(init_trans_legacy,
                                  dtype=o3c.Dtype.Float64,
                                  device=device)

        reg_p2p_t = o3d.t.pipelines.registration.registration_icp(
            source_t, target_t, max_correspondence_distance, init_trans_t,
            o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2))

        reg_p2p_legacy = o3d.pipelines.registration.registration_icp(
            source_legacy, target_legacy, max_correspondence_distance,
            init_trans_legacy,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

        np.testing.assert_allclose(reg_p2p_t.inlier_rmse,
                                   reg_p2p_legacy.inlier_rmse, 0.001)
        np.testing.assert_allclose(reg_p2p_t.fitness, reg_p2p_legacy.fitness,
                                   0.001)


@pytest.mark.parametrize("device", list_devices())
def test_test_registration_icp_point_to_plane(device):

    supported_dtypes = [o3c.Dtype.Float32, o3c.Dtype.Float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy_pointcloud()
        target_legacy = target_t.to_legacy_pointcloud()

        max_correspondence_distance = 1.5

        init_trans_legacy = np.array([[0.862, 0.011, -0.507, 0.5],
                                      [-0.139, 0.967, -0.215, 0.7],
                                      [0.487, 0.255, 0.835, -1.4],
                                      [0.0, 0.0, 0.0, 1.0]])
        init_trans_t = o3c.Tensor(init_trans_legacy,
                                  dtype=o3c.Dtype.Float64,
                                  device=device)

        reg_p2plane_t = o3d.t.pipelines.registration.registration_icp(
            source_t, target_t, max_correspondence_distance, init_trans_t,
            o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2))

        reg_p2plane_legacy = o3d.pipelines.registration.registration_icp(
            source_legacy, target_legacy, max_correspondence_distance,
            init_trans_legacy,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

        np.testing.assert_allclose(reg_p2plane_t.inlier_rmse,
                                   reg_p2plane_legacy.inlier_rmse, 0.001)
        np.testing.assert_allclose(reg_p2plane_t.fitness,
                                   reg_p2plane_legacy.fitness, 0.001)
