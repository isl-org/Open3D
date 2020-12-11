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

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
pytestmark = mltest.default_marks

# the supported dtypes for the values
dtypes = pytest.mark.parametrize('dtype',
                                 [np.int32, np.int64, np.float32, np.float64])

# this op is only available for torch


@dtypes
@mltest.parametrize.ml_torch_only
def test_ragged_to_dense(dtype, ml):

    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtype)
    row_splits = np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.int64)
    out_col_size = 4
    default_value = np.array(-1, dtype=dtype)

    ans = mltest.run_op(ml, ml.device, True, ml.ops.ragged_to_dense, values,
                        row_splits, out_col_size, default_value)

    expected = np.full((row_splits.shape[0] - 1, out_col_size), default_value)
    for i in range(row_splits.shape[0] - 1):
        for j, value_idx in enumerate(range(row_splits[i], row_splits[i + 1])):
            if j < expected.shape[1]:
                expected[i, j] = values[value_idx]

    np.testing.assert_equal(ans, expected)


# test with more dimensions
@dtypes
@mltest.parametrize.ml_torch_only
def test_ragged_to_dense_more_dims(dtype, ml):

    values = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                       [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
                      dtype=dtype)
    row_splits = np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.int64)
    out_col_size = 4
    default_value = np.array([-1, -1], dtype=dtype)

    ans = mltest.run_op(ml, ml.device, True, ml.ops.ragged_to_dense, values,
                        row_splits, out_col_size, default_value)

    expected = np.full((
        row_splits.shape[0] - 1,
        out_col_size,
    ) + default_value.shape, default_value)
    for i in range(row_splits.shape[0] - 1):
        for j, value_idx in enumerate(range(row_splits[i], row_splits[i + 1])):
            if j < expected.shape[1]:
                expected[i, j] = values[value_idx]

    np.testing.assert_equal(ans, expected)


# test with larger random data
@dtypes
@mltest.parametrize.ml_torch_only
@pytest.mark.parametrize('seed', [123, 456])
def test_ragged_to_dense_random(dtype, ml, seed):

    rng = np.random.RandomState(seed)

    values = rng.random(size=(10000,)).astype(dtype)
    row_splits = [0]
    while row_splits[-1] < values.shape[0]:
        row_splits.append(row_splits[-1] + rng.randint(0, 10))
    row_splits[-1] = values.shape[0]
    row_splits = np.array(row_splits, dtype=np.int64)
    out_col_size = rng.randint(1, 37)

    default_value = np.array(-1, dtype=dtype)

    ans = mltest.run_op(ml, ml.device, True, ml.ops.ragged_to_dense, values,
                        row_splits, out_col_size, default_value)

    expected = np.full((row_splits.shape[0] - 1, out_col_size), default_value)
    for i in range(row_splits.shape[0] - 1):
        for j, value_idx in enumerate(range(row_splits[i], row_splits[i + 1])):
            if j < expected.shape[1]:
                expected[i, j] = values[value_idx]

    np.testing.assert_equal(ans, expected)
