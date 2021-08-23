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
import numpy as np
import pytest
import mltest
import torch

# skip all tests if the tf ops were not built and disable warnings caused by
# tensorflow
pytestmark = mltest.default_marks

# the supported dtypes for the values
dtypes = pytest.mark.parametrize('dtype',
                                 [np.int32, np.int64, np.float32, np.float64])

# this class is only available for torch


@dtypes
@mltest.parametrize.ml_torch_only
def test_creation(dtype, ml):
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=dtype)
    row_splits = np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.int64)

    # From numpy arrays
    r_tensor = ml.classes.RaggedTensor.from_row_splits(values, row_splits)
    for i, tensor in enumerate(r_tensor):
        np.testing.assert_equal(mltest.to_numpy(tensor),
                                values[row_splits[i]:row_splits[i + 1]])

    # From List
    r_tensor = ml.classes.RaggedTensor.from_row_splits(list(values),
                                                       list(row_splits))
    for i, tensor in enumerate(r_tensor):
        np.testing.assert_equal(mltest.to_numpy(tensor),
                                values[row_splits[i]:row_splits[i + 1]])

    # Incompatible tensors.
    # Non zero first element.
    row_splits = np.array([1, 2, 4, 4, 5, 12, 13], dtype=np.int64)
    with np.testing.assert_raises(RuntimeError):
        ml.classes.RaggedTensor.from_row_splits(values, row_splits)

    # Rank > 1.
    row_splits = np.array([[0, 2, 4, 4, 5, 12, 13]], dtype=np.int64)
    with np.testing.assert_raises(RuntimeError):
        ml.classes.RaggedTensor.from_row_splits(values, row_splits)

    # Not increasing monotonically.
    row_splits = np.array([[0, 2, 4, 6, 5, 12, 13]], dtype=np.int64)
    with np.testing.assert_raises(RuntimeError):
        ml.classes.RaggedTensor.from_row_splits(values, row_splits)

    # Wrong dtype.
    row_splits = np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.float32)
    with np.testing.assert_raises(RuntimeError):
        ml.classes.RaggedTensor.from_row_splits(values, row_splits)


# test with more dimensions
@dtypes
@mltest.parametrize.ml_torch_only
def test_creation_more_dims(dtype, ml):
    values = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                       [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
                      dtype=dtype)
    row_splits = np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.int64)

    # From numpy arrays
    r_tensor = ml.classes.RaggedTensor.from_row_splits(values, row_splits)
    for i, tensor in enumerate(r_tensor):
        np.testing.assert_equal(mltest.to_numpy(tensor),
                                values[row_splits[i]:row_splits[i + 1]])

    # From List
    r_tensor = ml.classes.RaggedTensor.from_row_splits(list(values),
                                                       list(row_splits))
    for i, tensor in enumerate(r_tensor):
        np.testing.assert_equal(mltest.to_numpy(tensor),
                                values[row_splits[i]:row_splits[i + 1]])


@mltest.parametrize.ml_torch_only
def test_backprop(ml):
    # Create 3 different RaggedTensors and torch.tensor
    t_1 = torch.randn(10, 3, requires_grad=True)
    row_splits = torch.tensor([0, 4, 6, 6, 8, 10])
    r_1 = ml.classes.RaggedTensor.from_row_splits(t_1.detach().numpy(),
                                                  row_splits)
    r_1.requires_grad = True

    t_2 = torch.randn(10, 3, requires_grad=True)
    r_2 = ml.classes.RaggedTensor.from_row_splits(t_2.detach().numpy(),
                                                  row_splits)
    r_2.requires_grad = True

    t_3 = torch.randn(10, 3, requires_grad=True)
    r_3 = ml.classes.RaggedTensor.from_row_splits(t_3.detach().numpy(),
                                                  row_splits)
    r_3.requires_grad = True

    r_ans = (r_1 + r_2) * r_3
    t_ans = (t_1 + t_2) * t_3

    np.testing.assert_equal(mltest.to_numpy(t_ans),
                            mltest.to_numpy(r_ans.values))

    # Compute gradients
    t_ans.sum().backward()
    r_ans.values.sum().backward()

    np.testing.assert_equal(mltest.to_numpy(t_1.grad),
                            mltest.to_numpy(r_1.values.grad))


@dtypes
@mltest.parametrize.ml_torch_only
def test_binary_ew_ops(dtype, ml):
    # Binary Ops.
    t_1 = torch.from_numpy(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 dtype=dtype)).to(ml.device)
    t_2 = torch.from_numpy(
        np.array([2, 3, 6, 3, 11, 3, 43, 12, 8, 15, 12, 87, 45],
                 dtype=dtype)).to(ml.device)
    row_splits = torch.from_numpy(
        np.array([0, 2, 4, 4, 5, 12, 13], dtype=np.int64)).to(ml.device)

    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    b = ml.classes.RaggedTensor.from_row_splits(t_2, row_splits)

    np.testing.assert_equal(
        (a + b).values.cpu().numpy(),
        np.array([2, 4, 8, 6, 15, 8, 49, 19, 16, 24, 22, 98, 57]))
    np.testing.assert_equal(
        (a - b).values.cpu().numpy(),
        np.array([-2, -2, -4, 0, -7, 2, -37, -5, 0, -6, -2, -76, -33]))
    np.testing.assert_equal(
        (a * b).values.cpu().numpy(),
        np.array([0, 3, 12, 9, 44, 15, 258, 84, 64, 135, 120, 957, 540]))
    np.testing.assert_equal((a / b).values.cpu().numpy(),
                            (t_1 / t_2).cpu().numpy())
    np.testing.assert_equal((a // b).values.cpu().numpy(),
                            np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]))

    # Assignment Ops.
    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    a += b
    np.testing.assert_equal(
        a.values.cpu().numpy(),
        np.array([2, 4, 8, 6, 15, 8, 49, 19, 16, 24, 22, 98, 57]))

    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    a -= b
    np.testing.assert_equal(
        a.values.cpu().numpy(),
        np.array([-2, -2, -4, 0, -7, 2, -37, -5, 0, -6, -2, -76, -33]))

    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    a *= b
    np.testing.assert_equal(
        a.values.cpu().numpy(),
        np.array([0, 3, 12, 9, 44, 15, 258, 84, 64, 135, 120, 957, 540]))

    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    a //= b
    np.testing.assert_equal(a.values.cpu().numpy(),
                            np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]))

    # Failure cases with incompatible shape.
    # Different row_splits.
    row_splits = [0, 4, 5, 13]
    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    row_splits = [0, 4, 6, 13]
    b = ml.classes.RaggedTensor.from_row_splits(t_2, row_splits)

    with np.testing.assert_raises(ValueError):
        a + b
    with np.testing.assert_raises(ValueError):
        a += b

    # Different length
    row_splits = [0, 4, 5, 13]
    a = ml.classes.RaggedTensor.from_row_splits(t_1, row_splits)
    row_splits = [0, 4, 13]
    b = ml.classes.RaggedTensor.from_row_splits(t_2, row_splits)

    with np.testing.assert_raises(ValueError):
        a + b
    with np.testing.assert_raises(ValueError):
        a += b
