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

import numpy as np
import open3d as o3d
import pytest
import importlib


@pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                    reason='tf ops not built')
def test_tf_neighbors():
    ops = importlib.import_module('open3d.ml.tf.ops')

    query_points = np.array(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=np.float32)
    dataset_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=np.float32)
    radius = 0.1

    indices = ops.ordered_neighbors(query_points, dataset_points,
                                    radius).cpu().numpy()
    indices_ref = np.array([[1, 4], [1, 4]], dtype=np.int32)

    np.testing.assert_equal(indices, indices_ref)

    indices = ops.ordered_neighbors(query_points, dataset_points,
                                    0.2).cpu().numpy()
    indices_ref = np.array(
        [[1, 4, 9, 0, 3, 2, 5, 7, 6], [1, 4, 9, 0, 3, 2, 5, 7, 6]],
        dtype=np.int32)

    np.testing.assert_equal(indices, indices_ref)

    with pytest.raises(ValueError):
        ops.ordered_neighbors(None, dataset_points, 0.1)

    with pytest.raises(ValueError):
        ops.ordered_neighbors(query_points, None, 1)


@pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                    reason='tf ops not built')
def test_tf_batch_neighbors():
    ops = importlib.import_module('open3d.ml.tf.ops')

    query_points = np.array(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843],
         [0.064705, 0.043921, 0.087843]],
        dtype=np.float32)
    dataset_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1],
         [0.0, 0.0, 0.2], [0.0, 0.1, 0.0], [0.0, 0.1, 0.1], [0.0, 0.1, 0.2],
         [0.0, 0.2, 0.0], [0.0, 0.2, 0.1], [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=np.float32)
    radius = 0.1

    indices = ops.batch_ordered_neighbors(query_points, dataset_points, [1, 2],
                                          [10, 10], radius)
    indices_ref = np.array([[1, 4], [11, 14], [11, 14]], dtype=np.int32)

    np.testing.assert_equal(indices, indices_ref)

    assert indices.dtype == np.int32

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)

    indices = ops.batch_ordered_neighbors(points, points, [2, 3, 2], [3, 2, 2],
                                          11).cpu().numpy()

    indices_ref = np.array([[0, 1, 2], [1, 0, 2], [3, 4, 7], [3, 4, 7],
                            [4, 3, 7], [5, 6, 7], [6, 5, 7]],
                           dtype=np.int32)
    np.testing.assert_equal(indices, indices_ref)

    indices = ops.batch_ordered_neighbors(points, points, [1, 1, 5], [5, 1, 1],
                                          11).cpu().numpy()

    indices_ref = np.ones((7, 5), dtype=np.int32) * 7
    indices_ref[0] = [0, 1, 2, 3, 4]
    indices_ref[:, 0] = [0, 5, 6, 6, 6, 6, 6]
    np.testing.assert_equal(indices, indices_ref)

    with pytest.raises(ValueError):
        ops.batch_ordered_neighbors(None, dataset_points, [1, 2], [2, 1], 0.1)

    with pytest.raises(ValueError):
        ops.batch_ordered_neighbors(query_points, None, [1, 2], [2, 1], 1)

    with pytest.raises(ValueError):
        ops.batch_ordered_neighbors(query_points, dataset_points, None, [2, 1],
                                    1)
