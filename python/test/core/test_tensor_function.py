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
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


def list_dtypes():
    return [
        o3c.float32,
        o3c.float64,
        o3c.int8,
        o3c.int16,
        o3c.int32,
        o3c.int64,
        o3c.uint8,
        o3c.uint16,
        o3c.uint32,
        o3c.uint64,
        o3c.bool,
    ]


def list_non_bool_dtypes():
    return [
        o3c.float32,
        o3c.float64,
        o3c.int8,
        o3c.int16,
        o3c.int32,
        o3c.int64,
        o3c.uint8,
        o3c.uint16,
        o3c.uint32,
        o3c.uint64,
    ]


@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_concatenate(dtype, device):

    # 0-D cannot be concatenated.
    a = o3c.Tensor(0, dtype=dtype, device=device)
    b = o3c.Tensor(0, dtype=dtype, device=device)
    c = o3c.Tensor(0, dtype=dtype, device=device)

    with pytest.raises(
            RuntimeError,
            match=r"Zero-dimensional tensor can only be concatenated along "
            "axis = null, but got 0."):
        o3c.concatenate((a, b, c))

    # Concatenating 1-D tensors.
    # 1-D can be concatenated along axis = 0, -1.
    a = o3c.Tensor([0, 1, 2], dtype=dtype, device=device)
    b = o3c.Tensor([3, 4], dtype=dtype, device=device)
    c = o3c.Tensor([5, 6, 7], dtype=dtype, device=device)

    # Default axis is 0.
    output_t = o3c.concatenate((a, b, c))
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()))

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.concatenate((a, b, c), axis=-1)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # So, 1-D tensors cannot be concatenated along axis 1 or -2.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 1, but it must satisfy -1 <= dim <= 0"
    ):
        o3c.concatenate((a, b, c), axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -2, but it must satisfy -1 <= dim <= 0"
    ):
        o3c.concatenate((a, b, c), axis=-2)

    # Concatenating 2-D tensors.
    a = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    b = o3c.Tensor([[4, 5]], dtype=dtype, device=device)
    c = o3c.Tensor([[6, 7]], dtype=dtype, device=device)

    # Above 2-D tensors can be concatenated along axis = 0, -2.
    # Default axis is 0.
    output_t = o3c.concatenate((a, b, c))
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()))

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.concatenate((a, b, c), axis=-2)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # Above 2-D tensors cannot be appended to 2-D along axis = 1, -1.
    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        o3c.concatenate((a, b, c), axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        o3c.concatenate((a, b, c), axis=-1)

    # Concatenating 2-D tensors of shape {3, 1}.
    a = o3c.Tensor([[0], [1], [2]], dtype=dtype, device=device)
    b = o3c.Tensor([[3], [4], [5]], dtype=dtype, device=device)
    c = o3c.Tensor([[6], [7], [8]], dtype=dtype, device=device)

    # Above 2-D tensors can be concatenated along axis = 0, 1, -1, -2.
    output_t = o3c.concatenate((a, b, c), axis=0)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.concatenate((a, b, c), axis=1)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.concatenate((a, b, c), axis=-1)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.concatenate((a, b, c), axis=-2)
    output_np = np.concatenate(
        (a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()), axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # Above 2-D tensors cannot be appended to 2-D along axis = 2, -3.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 2, but it must satisfy -2 <= dim <= 1"
    ):
        o3c.concatenate((a, b, c), axis=2)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -3, but it must satisfy -2 <= dim <= 1"
    ):
        o3c.concatenate((a, b, c), axis=-3)

    # Using concatenate for a single tensor. The tensor is split along its
    # first dimension, and concatenated along the axis.
    a = o3c.Tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
                   dtype=o3c.Dtype.Float32,
                   device=device)
    output_t = o3c.concatenate((a), axis=1)
    output_np = np.concatenate((a.cpu().numpy()), axis=1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # dtype and device must be same for all the input tensors.
    a = o3c.Tensor([[0, 1], [2, 3]], dtype=o3c.Dtype.Float32, device=device)
    b = o3c.Tensor([[4, 5]], dtype=o3c.Dtype.Float64, device=device)
    with pytest.raises(
            RuntimeError,
            match=r"Tensor has dtype Float64, but is expected to have Float32"):
        o3c.concatenate((a, b))


@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_append(dtype, device):
    # Appending 0-D.
    # 0-D can only be appended along axis = null.
    self = o3c.Tensor(0, dtype=dtype, device=device)
    values = o3c.Tensor(1, dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    with pytest.raises(
            RuntimeError,
            match=r"Zero-dimensional tensor can only be concatenated along "
            "axis = null, but got 0."):
        o3c.append(self=self, values=values, axis=0)

    # Appending 1-D.
    # 1-D can be appended along axis = 0, -1.
    self = o3c.Tensor([0, 1], dtype=dtype, device=device)
    values = o3c.Tensor([2, 3, 4], dtype=dtype, device=device)
    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # axis must always be in range [-num_dims, num_dims).
    # So, 1-D tensor cannot be appended along axis 1 or -2.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 1, but it must satisfy -1 <= dim <= 0"
    ):
        o3c.append(self=self, values=values, axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -2, but it must satisfy -1 <= dim <= 0"
    ):
        o3c.append(self=self, values=values, axis=-2)

    # Appending 2-D. [2, 2] to [2, 2].
    # [2, 2] to [2, 2] can be appended along axis = 0, 1, -1, -2.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5], [6, 7]], dtype=dtype, device=device)

    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # axis must always be in range [-num_dims, num_dims).
    # So, 2-D tensor cannot be appended along axis 2 or -3.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 2, but it must satisfy -2 <= dim <= 1"
    ):
        o3c.append(self=self, values=values, axis=2)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -3, but it must satisfy -2 <= dim <= 1"
    ):
        o3c.append(self=self, values=values, axis=-3)

    # Appending 2-D. [1, 2] to [2, 2].
    # [1, 2] to [2, 2] can be appended along axis = 0, -2.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5]], dtype=dtype, device=device)

    output_t = o3c.append(self=self, values=values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = o3c.append(self=self, values=values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # all the dimensions other than the dimension along the axis, must be
    # exactly same.
    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        o3c.append(self=self, values=values, axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        o3c.append(self=self, values=values, axis=-1)

    # dtype and device must be same for all the input tensors.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=o3c.Dtype.Float32, device=device)
    values = o3c.Tensor([[4, 5]], dtype=o3c.Dtype.Float64, device=device)
    with pytest.raises(
            RuntimeError,
            match=r"Tensor has dtype Float64, but is expected to have Float32"):
        o3c.append(self=self, values=values)
