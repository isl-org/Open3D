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


def get_pcds_and_correspondences(dtype, device):
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

    source_t.point.positions = source_points
    target_t.point.positions = target_points
    target_t.point.normals = target_normals

    correspondences = o3c.Tensor([10, 1, 1, 3, 2, 5, 9, 7, 5, 8, 7, 7, 5, 8],
                                 o3c.int64, device)

    return source_t, target_t, correspondences


@pytest.mark.parametrize("device", list_devices())
def test_compute_rmse_point_to_point(device):

    supported_dtypes = [o3c.Dtype.Float32, o3c.Dtype.Float64]
    for dtype in supported_dtypes:
        source_t, target_t, correspondences = get_pcds_and_correspondences(
            dtype, device)

        estimation_p2p = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(
        )
        p2p_rmse = estimation_p2p.compute_rmse(source_t, target_t,
                                               correspondences)

        np.testing.assert_allclose(p2p_rmse, 0.706437, 0.0001)


@pytest.mark.parametrize("device", list_devices())
def test_compute_transformation_point_to_point(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t, correspondences = get_pcds_and_correspondences(
            dtype, device)

        estimation_p2p = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(
        )

        transformation_p2p = estimation_p2p.compute_transformation(
            source_t, target_t, correspondences)
        source_transformed_p2p = source_t.transform(
            transformation_p2p.to(device, dtype))
        p2p_rmse = estimation_p2p.compute_rmse(source_transformed_p2p, target_t,
                                               correspondences)

        np.testing.assert_allclose(p2p_rmse, 0.578255, 0.0001)


@pytest.mark.parametrize("device", list_devices())
def test_compute_rmse_point_to_plane(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t, correspondences = get_pcds_and_correspondences(
            dtype, device)

        estimation_p2l = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(
        )
        p2l_rmse = estimation_p2l.compute_rmse(source_t, target_t,
                                               correspondences)

        np.testing.assert_allclose(p2l_rmse, 0.335499, 0.0001)


@pytest.mark.parametrize("device", list_devices())
def test_compute_transformation_point_to_plane(device):

    supported_dtypes = [o3c.float32, o3c.float64]
    for dtype in supported_dtypes:
        source_t, target_t, correspondences = get_pcds_and_correspondences(
            dtype, device)

        estimation_p2l = o3d.t.pipelines.registration.TransformationEstimationPointToPlane(
        )

        transformation_p2l = estimation_p2l.compute_transformation(
            source_t, target_t, correspondences)
        source_transformed_p2l = source_t.transform(
            transformation_p2l.to(device, dtype))

        p2l_rmse = estimation_p2l.compute_rmse(source_transformed_p2l, target_t,
                                               correspondences)

        np.testing.assert_allclose(p2l_rmse, 0.601422, 0.0001)
