# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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
    dtype = o3c.Dtype.Float32

    source_points = o3c.Tensor(
        [[1.15495, 2.40671, 1.15061], [1.81481, 2.06281, 1.71927],
         [0.888322, 2.05068, 2.04879], [3.78842, 1.70788, 1.30246],
         [1.8437, 2.22894, 0.986237], [2.95706, 2.2018, 0.987878],
         [1.72644, 1.24356, 1.93486], [0.922024, 1.14872, 2.34317],
         [3.70293, 1.85134, 1.15357], [3.06505, 1.30386, 1.55279],
         [0.634826, 1.04995, 2.47046], [1.40107, 1.37469, 1.09687],
         [2.93002, 1.96242, 1.48532], [3.74384, 1.30258, 1.30244]], dtype,
        device)

    target_points = o3c.Tensor(
        [[2.41766, 2.05397, 1.74994], [1.37848, 2.19793, 1.66553],
         [2.24325, 2.27183, 1.33708], [3.09898, 1.98482, 1.77401],
         [1.81615, 1.48337, 1.49697], [3.01758, 2.20312, 1.51502],
         [2.38836, 1.39096, 1.74914], [1.30911, 1.4252, 1.37429],
         [3.16847, 1.39194, 1.90959], [1.59412, 1.53304, 1.5804],
         [1.34342, 2.19027, 1.30075]], dtype, device)

    target_normals = o3c.Tensor(
        [[-0.0085016, -0.22355, -0.519574], [0.257463, -0.0738755, -0.698319],
         [0.0574301, -0.484248, -0.409929], [-0.0123503, -0.230172, -0.52072],
         [0.355904, -0.142007, -0.720467], [0.0674038, -0.418757, -0.458602],
         [0.226091, 0.258253, -0.874024], [0.43979, 0.122441, -0.574998],
         [0.109144, 0.180992, -0.762368], [0.273325, 0.292013, -0.903111],
         [0.385407, -0.212348, -0.277818]], dtype, device)

    source_t = o3d.t.geometry.PointCloud(device)
    target_t = o3d.t.geometry.PointCloud(device)

    source_t.point["points"] = source_points
    target_t.point["points"] = target_points
    target_t.point["normals"] = target_normals

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
                               evaluation_legacy.inlier_rmse, 0.0001)
    np.testing.assert_allclose(evaluation_t.fitness, evaluation_legacy.fitness,
                               0.0001)


@pytest.mark.parametrize("device", list_devices())
def test_registration_icp_point_to_point(device):
    dtype = o3c.Dtype.Float32

    source_points = o3c.Tensor(
        [[1.15495, 2.40671, 1.15061], [1.81481, 2.06281, 1.71927],
         [0.888322, 2.05068, 2.04879], [3.78842, 1.70788, 1.30246],
         [1.8437, 2.22894, 0.986237], [2.95706, 2.2018, 0.987878],
         [1.72644, 1.24356, 1.93486], [0.922024, 1.14872, 2.34317],
         [3.70293, 1.85134, 1.15357], [3.06505, 1.30386, 1.55279],
         [0.634826, 1.04995, 2.47046], [1.40107, 1.37469, 1.09687],
         [2.93002, 1.96242, 1.48532], [3.74384, 1.30258, 1.30244]], dtype,
        device)

    target_points = o3c.Tensor(
        [[2.41766, 2.05397, 1.74994], [1.37848, 2.19793, 1.66553],
         [2.24325, 2.27183, 1.33708], [3.09898, 1.98482, 1.77401],
         [1.81615, 1.48337, 1.49697], [3.01758, 2.20312, 1.51502],
         [2.38836, 1.39096, 1.74914], [1.30911, 1.4252, 1.37429],
         [3.16847, 1.39194, 1.90959], [1.59412, 1.53304, 1.5804],
         [1.34342, 2.19027, 1.30075]], dtype, device)

    source_t = o3d.t.geometry.PointCloud(device)
    target_t = o3d.t.geometry.PointCloud(device)

    source_t.point["points"] = source_points
    target_t.point["points"] = target_points

    source_legacy = source_t.to_legacy_pointcloud()
    target_legacy = target_t.to_legacy_pointcloud()

    max_correspondence_distance = 1.25

    init_trans_legacy = np.eye(4)
    init_trans_t = o3c.Tensor.eye(4, o3c.Dtype.Float64, device)

    reg_p2p_t = o3d.t.pipelines.registration.registration_icp(
        source_t, target_t, max_correspondence_distance, init_trans_t,
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

    reg_p2p_legacy = o3d.pipelines.registration.registration_icp(
        source_legacy, target_legacy, max_correspondence_distance,
        init_trans_legacy,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

    np.testing.assert_allclose(reg_p2p_t.inlier_rmse,
                               reg_p2p_legacy.inlier_rmse, 0.0001)
    np.testing.assert_allclose(reg_p2p_t.fitness, reg_p2p_legacy.fitness,
                               0.0001)


@pytest.mark.parametrize("device", list_devices())
def test_test_registration_icp_point_to_plane(device):
    dtype = o3c.Dtype.Float32

    source_points = o3c.Tensor(
        [[1.15495, 2.40671, 1.15061], [1.81481, 2.06281, 1.71927],
         [0.888322, 2.05068, 2.04879], [3.78842, 1.70788, 1.30246],
         [1.8437, 2.22894, 0.986237], [2.95706, 2.2018, 0.987878],
         [1.72644, 1.24356, 1.93486], [0.922024, 1.14872, 2.34317],
         [3.70293, 1.85134, 1.15357], [3.06505, 1.30386, 1.55279],
         [0.634826, 1.04995, 2.47046], [1.40107, 1.37469, 1.09687],
         [2.93002, 1.96242, 1.48532], [3.74384, 1.30258, 1.30244]], dtype,
        device)

    target_points = o3c.Tensor(
        [[2.41766, 2.05397, 1.74994], [1.37848, 2.19793, 1.66553],
         [2.24325, 2.27183, 1.33708], [3.09898, 1.98482, 1.77401],
         [1.81615, 1.48337, 1.49697], [3.01758, 2.20312, 1.51502],
         [2.38836, 1.39096, 1.74914], [1.30911, 1.4252, 1.37429],
         [3.16847, 1.39194, 1.90959], [1.59412, 1.53304, 1.5804],
         [1.34342, 2.19027, 1.30075]], dtype, device)

    target_normals = o3c.Tensor(
        [[-0.0085016, -0.22355, -0.519574], [0.257463, -0.0738755, -0.698319],
         [0.0574301, -0.484248, -0.409929], [-0.0123503, -0.230172, -0.52072],
         [0.355904, -0.142007, -0.720467], [0.0674038, -0.418757, -0.458602],
         [0.226091, 0.258253, -0.874024], [0.43979, 0.122441, -0.574998],
         [0.109144, 0.180992, -0.762368], [0.273325, 0.292013, -0.903111],
         [0.385407, -0.212348, -0.277818]], dtype, device)

    source_t = o3d.t.geometry.PointCloud(device)
    target_t = o3d.t.geometry.PointCloud(device)

    source_t.point["points"] = source_points
    target_t.point["points"] = target_points
    target_t.point["normals"] = target_normals

    source_legacy = source_t.to_legacy_pointcloud()
    target_legacy = target_t.to_legacy_pointcloud()

    max_correspondence_distance = 1.25
    init_trans_legacy = np.eye(4)
    init_trans_t = o3c.Tensor.eye(4, o3c.Dtype.Float64, device)

    reg_p2plane_t = o3d.t.pipelines.registration.registration_icp(
        source_t, target_t, max_correspondence_distance, init_trans_t,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

    reg_p2plane_legacy = o3d.pipelines.registration.registration_icp(
        source_legacy, target_legacy, max_correspondence_distance,
        init_trans_legacy,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2))

    np.testing.assert_allclose(reg_p2plane_t.inlier_rmse,
                               reg_p2plane_legacy.inlier_rmse, 0.0001)
    np.testing.assert_allclose(reg_p2plane_t.fitness,
                               reg_p2plane_legacy.fitness, 0.0001)
