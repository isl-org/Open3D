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
import numpy as np
import pytest
import mltest
from check_gradients import check_gradients

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks

# the supported dtypes
position_dtypes = pytest.mark.parametrize('pos_dtype', [np.float32, np.float64])
feature_dtypes = pytest.mark.parametrize(
    'feat_dtype', [np.float32, np.float64, np.int32, np.int64])

# aggregation functions
position_functions = pytest.mark.parametrize(
    'position_fn', ['average', 'center', 'nearest_neighbor'])
feature_functions = pytest.mark.parametrize(
    'feature_fn', ['average', 'max', 'nearest_neighbor'])


@mltest.parametrize.ml_cpu_only
@position_dtypes
@feature_dtypes
@position_functions
@feature_functions
def test_voxel_pooling(ml, pos_dtype, feat_dtype, position_fn, feature_fn):
    # yapf: disable

    points = np.array([
        # 3 points in voxel
        [0.5, 0.5, 0.5],
        [0.7, 0.2, 0.3],
        [0.7, 0.5, 0.9],
        # 2 points in another voxel
        [1.4, 1.5, 1.4],
        [1.7, 1.2, 1.3],
        ], dtype=pos_dtype)

    features = np.array([
        # 3 points in voxel
        [1,1],
        [2,1],
        [3,1],
        # 2 points in another voxel
        [4,1],
        [5,1],
        ], dtype=feat_dtype)

    # yapf: enable

    voxel_size = 1
    ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling, points,
                        features, voxel_size, position_fn, feature_fn)

    if position_fn == 'average':
        expected_positions = np.stack(
            [np.mean(points[:3], axis=0),
             np.mean(points[3:], axis=0)])
    elif position_fn == 'center':
        expected_positions = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
                                      dtype=pos_dtype)
    elif position_fn == 'nearest_neighbor':
        expected_positions = np.array([points[0], points[3]], dtype=pos_dtype)

    assert len(ans.pooled_positions) == 2

    # compute assignment
    if np.linalg.norm(ans.pooled_positions[0] -
                      expected_positions[0]) < np.linalg.norm(
                          ans.pooled_positions[0] - expected_positions[1]):
        index = [0, 1]
    else:
        index = [1, 0]

    np.testing.assert_allclose(ans.pooled_positions, expected_positions[index])

    if feature_fn == 'average':
        if np.issubdtype(feat_dtype, np.integer):
            expected_features = np.stack([
                np.sum(features[:3], axis=0) // 3,
                np.sum(features[3:], axis=0) // 2
            ])
        else:
            expected_features = np.stack(
                [np.mean(features[:3], axis=0),
                 np.mean(features[3:], axis=0)])
    elif feature_fn == 'max':
        expected_features = np.stack(
            [np.max(features[:3], axis=0),
             np.max(features[3:], axis=0)])
    elif feature_fn == 'nearest_neighbor':
        expected_features = np.array([features[0], features[3]])

    np.testing.assert_allclose(ans.pooled_features, expected_features[index])


@mltest.parametrize.ml_cpu_only
@position_dtypes
@feature_dtypes
@position_functions
@feature_functions
def test_voxel_pooling_empty_point_set(ml, pos_dtype, feat_dtype, position_fn,
                                       feature_fn):
    points = np.zeros(shape=[0, 3], dtype=pos_dtype)
    features = np.zeros(shape=[0, 5], dtype=feat_dtype)

    voxel_size = 1
    ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling, points,
                        features, voxel_size, position_fn, feature_fn)

    np.testing.assert_array_equal(points, ans.pooled_positions)
    np.testing.assert_array_equal(features, ans.pooled_features)


# tf and torch does not support gradient computation for integer types
gradient_feature_dtypes = pytest.mark.parametrize('feat_dtype',
                                                  [np.float32, np.float64])


@mltest.parametrize.ml_cpu_only
@position_dtypes
@gradient_feature_dtypes
@position_functions
@feature_functions
@pytest.mark.parametrize('empty_point_set', [
    False,
])
def test_voxel_pooling_grad(ml, pos_dtype, feat_dtype, position_fn, feature_fn,
                            empty_point_set):

    rng = np.random.RandomState(123)

    N = 0 if empty_point_set else 50
    channels = 4
    positions = rng.uniform(0, 1, (N, 3)).astype(pos_dtype)

    # generate features and make sure that the feature values are not too close to each other
    # if they are too close the numerical jacobian will be wrong
    features = np.linspace(0, N * channels, num=N * channels, endpoint=False)
    np.random.shuffle(features)
    features = np.reshape(features, (N, channels)).astype(feat_dtype)
    voxel_size = 0.25

    def fn(features):
        ans = mltest.run_op(ml, ml.device, True, ml.ops.voxel_pooling,
                            positions, features, voxel_size, position_fn,
                            feature_fn)
        return ans.pooled_features

    def fn_grad(features_bp, features):
        return mltest.run_op_grad(ml, ml.device, True, ml.ops.voxel_pooling,
                                  features, 'pooled_features', features_bp,
                                  positions, features, voxel_size, position_fn,
                                  feature_fn)

    gradient_OK = check_gradients(features, fn, fn_grad, epsilon=1)
    assert gradient_OK
