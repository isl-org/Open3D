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

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks

# the supported input dtypes
value_dtypes = pytest.mark.parametrize(
    'dtype', [np.int32, np.int64, np.float32, np.float64])


@pytest.mark.parametrize('seed', range(3))
@value_dtypes
@mltest.parametrize.ml
def test_reduce_subarrays_sum_random(seed, dtype, ml):

    rng = np.random.RandomState(seed)

    values_shape = [rng.randint(100, 200)]
    values = rng.uniform(0, 10, size=values_shape).astype(dtype)

    row_splits = [0]
    for _ in range(rng.randint(1, 10)):
        row_splits.append(
            rng.randint(0, values_shape[0] - row_splits[-1]) + row_splits[-1])
    row_splits.extend(values_shape)

    expected_result = []
    for start, stop in zip(row_splits, row_splits[1:]):
        # np.sum correctly handles zero length arrays and returns 0
        expected_result.append(np.sum(values[start:stop]))
    np.array(expected_result, dtype=dtype)

    row_splits = np.array(row_splits, dtype=np.int64)

    ans = mltest.run_op(ml,
                        ml.device,
                        True,
                        ml.ops.reduce_subarrays_sum,
                        values=values,
                        row_splits=row_splits)

    if np.issubdtype(dtype, np.integer):
        np.testing.assert_equal(ans, expected_result)
    else:  # floating point types
        np.testing.assert_allclose(ans, expected_result, rtol=1e-5, atol=1e-8)


@mltest.parametrize.ml
def test_reduce_subarrays_sum_zero_length_values(ml):

    rng = np.random.RandomState(1)

    shape = [rng.randint(100, 200)]
    values = np.array([], dtype=np.float32)

    row_splits = [0]
    for _ in range(rng.randint(1, 10)):
        row_splits.append(
            rng.randint(0, shape[0] - row_splits[-1]) + row_splits[-1])
    row_splits.extend(shape)
    row_splits = np.array(row_splits, dtype=np.int64)

    ans = mltest.run_op(ml,
                        ml.device,
                        True,
                        ml.ops.reduce_subarrays_sum,
                        values=values,
                        row_splits=row_splits)

    assert ans.shape == values.shape
    assert ans.dtype == values.dtype
