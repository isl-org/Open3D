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

from open3d_test import list_devices


def get_pcds(dtype, device):

    source_points = o3c.Tensor(
        [[1.0, 0.5, 2.0], [0.5, 0.5, 2.0], [0.5, 0.5, 2.5], [3.0, 1.0, 1.5],
         [3.5, 1.0, 1.0], [0.5, 1.0, 2.0], [1.5, 1.0, 1.5], [2.0, 1.0, 1.5],
         [1.0, 1.0, 2.0], [2.5, 1.0, 1.5], [3.0, 1.0, 1.0], [0.5, 1.0, 2.5],
         [1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [1.0, 1.5, 1.0], [3.0, 1.5, 1.0],
         [3.5, 1.5, 1.0], [3.0, 1.5, 1.5], [0.5, 1.5, 1.5], [0.5, 1.5, 2.0],
         [1.0, 1.5, 2.0], [2.5, 1.5, 1.5], [1.5, 1.5, 1.0], [1.5, 1.5, 1.5],
         [2.0, 1.5, 1.5], [3.0, 1.5, 0.5], [2.5, 1.5, 1.0], [2.0, 1.5, 1.0],
         [3.0, 2.0, 0.5], [0.5, 2.0, 1.5], [3.0, 2.0, 1.0], [1.0, 2.0, 1.0],
         [2.0, 1.5, 0.5], [0.5, 2.0, 2.0], [2.5, 2.0, 1.0], [2.5, 2.0, 0.5],
         [2.0, 2.0, 0.5], [2.5, 1.5, 0.5], [3.0, 2.0, 1.5], [2.0, 2.0, 1.0],
         [1.0, 2.0, 2.0], [1.5, 2.0, 1.0], [1.5, 2.0, 1.5], [2.5, 2.0, 1.5],
         [2.0, 2.0, 1.5], [1.0, 2.0, 0.5], [0.5, 2.0, 1.0], [1.5, 2.0, 0.5],
         [1.0, 2.0, 1.5]], dtype, device)

    target_points = o3c.Tensor(
        [[1.5, 1.0, 1.5], [2.5, 1.0, 1.5], [1.5, 1.0, 1.0], [1.0, 1.0, 1.0],
         [2.0, 1.0, 1.5], [3.0, 1.0, 1.5], [1.0, 1.0, 0.5], [1.0, 1.5, 1.0],
         [1.0, 1.5, 0.5], [1.0, 1.0, 1.5], [3.0, 1.0, 2.0], [3.0, 1.5, 2.0],
         [3.0, 1.5, 1.5], [1.0, 1.5, 1.5], [1.5, 1.5, 1.5], [2.5, 1.5, 1.5],
         [2.0, 1.5, 1.5], [1.5, 1.5, 1.0], [2.5, 1.5, 2.0], [1.0, 2.0, 1.0],
         [1.0, 2.0, 0.5], [2.5, 1.5, 1.0], [3.0, 2.0, 1.5], [2.5, 2.0, 1.0],
         [2.5, 2.0, 1.5], [1.5, 2.0, 1.0], [2.0, 1.5, 1.0], [1.0, 2.0, 1.5],
         [2.0, 2.0, 1.0], [1.5, 2.0, 1.5], [1.5, 2.0, 0.5], [2.0, 2.0, 1.5],
         [2.0, 2.0, 0.5], [1.5, 2.5, 1.0], [1.0, 2.5, 1.0], [3.0, 2.0, 1.0],
         [2.0, 2.5, 1.0], [2.5, 2.5, 1.0]], dtype, device)

    target_normals = o3c.Tensor(
        [[0.15597, -0.0463812, -0.986672], [-0.213545, 0.887963, 0.407334],
         [0.423193, -0.121977, -0.897792], [0.202251, 0.27611, -0.939605],
         [0.275452, 0.207216, -0.938716], [0.326146, 0.0385317, -0.944534],
         [0.983129, -0.174668, -0.0543011], [0.898665, -0.0602029, 0.434485],
         [0.711325, 0.193223, -0.675783], [0.346158, 0.198724, -0.916888],
         [0.302085, 0.28938, -0.908297], [0.341044, 0.414138, -0.843907],
         [0.212191, 0.213068, -0.953717], [0.239759, 0.313187, -0.918929],
         [0.302290, 0.27265, -0.913391], [0.209796, 0.402747, -0.890944],
         [0.267025, 0.218226, -0.938656], [0.00126928, -0.976587, -0.21512],
         [0.321912, 0.194736, -0.926526], [0.831227, 0.236675, -0.503037],
         [0.987006, -0.155324, 0.0411639], [0.103384, -0.808796, -0.57893],
         [0.181245, 0.66226, -0.727023], [0.235471, 0.525053, -0.817846],
         [0.231954, 0.446165, -0.864369], [-0.261931, -0.725542, -0.636381],
         [0.120953, -0.864985, -0.487003], [0.858345, -0.227847, 0.459706],
         [-0.416259, -0.367408, -0.831709], [-0.476652, 0.206048, -0.854604],
         [-0.211959, -0.523378, -0.825317], [-0.964914, 0.0541031, -0.256931],
         [-0.0653566, -0.913961, -0.400504], [-0.846868, -0.170805, -0.503628],
         [0.0366971, 0.515834, -0.855902], [-0.0714554, -0.855019, -0.513651],
         [-0.0217377, -0.957744, -0.286799], [-0.0345231, -0.947096, -0.319088]
        ], dtype, device)

    source = o3d.t.geometry.PointCloud(device)
    target = o3d.t.geometry.PointCloud(device)

    source.point["positions"] = source_points
    target.point["positions"] = target_points
    target.point["normals"] = target_normals

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
    dtype = o3c.float64

    # Constructor.
    registration_result = o3d.t.pipelines.registration.RegistrationResult()

    # Checking default values.
    assert registration_result.inlier_rmse == 0.0
    assert registration_result.fitness == 0.0
    assert registration_result.transformation.allclose(
        o3c.Tensor.eye(4, dtype, o3c.Device("CPU:0")))


@pytest.mark.parametrize("device", list_devices())
def test_evaluate_registration(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy()
        target_legacy = target_t.to_legacy()

        max_correspondence_distance = 3.0
        init_trans_legacy = np.eye(4)
        init_trans_t = o3c.Tensor.eye(4, o3c.float64, device)

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
def test_icp_point_to_point(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy()
        target_legacy = target_t.to_legacy()

        max_correspondence_distance = 3.0

        init_trans_legacy = np.array([[0.862, 0.011, -0.507, 0.5],
                                      [-0.139, 0.967, -0.215, 0.7],
                                      [0.487, 0.255, 0.835, -1.4],
                                      [0.0, 0.0, 0.0, 1.0]])
        init_trans_t = o3c.Tensor(init_trans_legacy,
                                  dtype=o3c.float64,
                                  device=device)

        reg_p2p_t = o3d.t.pipelines.registration.icp(
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
def test_icp_point_to_plane(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy()
        target_legacy = target_t.to_legacy()

        max_correspondence_distance = 3.0

        init_trans_legacy = np.array([[0.862, 0.011, -0.507, 0.5],
                                      [-0.139, 0.967, -0.215, 0.7],
                                      [0.487, 0.255, 0.835, -1.4],
                                      [0.0, 0.0, 0.0, 1.0]])
        init_trans_t = o3c.Tensor(init_trans_legacy,
                                  dtype=o3c.float64,
                                  device=device)

        reg_p2plane_t = o3d.t.pipelines.registration.icp(
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


@pytest.mark.parametrize("device", list_devices())
def test_get_information_matrix(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t = get_pcds(dtype, device)

        source_legacy = source_t.to_legacy()
        target_legacy = target_t.to_legacy()

        max_correspondence_distance = 3.0

        transformation_legacy = np.array([[0.862, 0.011, -0.507, 0.5],
                                          [-0.139, 0.967, -0.215, 0.7],
                                          [0.487, 0.255, 0.835, -1.4],
                                          [0.0, 0.0, 0.0, 1.0]])
        transformation_t = o3c.Tensor(transformation_legacy,
                                      dtype=o3c.float64,
                                      device=device)

        info_matrix_t = o3d.t.pipelines.registration.get_information_matrix(
            source_t, target_t, max_correspondence_distance, transformation_t)

        info_matrix_legacy = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_legacy, target_legacy, max_correspondence_distance,
            transformation_legacy)

        np.testing.assert_allclose(info_matrix_t.cpu().numpy(),
                                   info_matrix_legacy, 1e-1, 1e-1)
