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
import core_test_utils


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_tensorlist_create(device):
    dtype = o3c.Dtype.Float32

    # Construct uninitialized tensorlist.
    tl = o3c.TensorList(o3c.SizeVector([3, 4]), dtype, device)
    assert tl.size == 0
    assert tl.element_shape == o3c.SizeVector([3, 4])

    tl = o3c.TensorList(o3c.SizeVector([3, 4]), dtype, device=device)
    assert tl.size == 0
    assert tl.element_shape == o3c.SizeVector([3, 4])

    tl = o3c.TensorList(o3c.SizeVector([3, 4]), dtype=dtype, device=device)
    assert tl.size == 0
    assert tl.element_shape == o3c.SizeVector([3, 4])

    tl = o3c.TensorList(element_shape=o3c.SizeVector([3, 4]),
                        dtype=dtype,
                        device=device)
    assert tl.size == 0
    assert tl.element_shape == o3c.SizeVector([3, 4])

    # Construct from a list of tensors.
    t0 = o3c.Tensor.ones((2, 3), dtype, device) * 0
    t1 = o3c.Tensor.ones((2, 3), dtype, device) * 1
    t2 = o3c.Tensor.ones((2, 3), dtype, device) * 2
    tl = o3c.TensorList([t0, t1, t2])
    assert tl[0].allclose(t0)
    assert tl[1].allclose(t1)
    assert tl[2].allclose(t2)
    assert tl[-1].allclose(t2)
    assert not tl[0].issame(t0)
    assert not tl[1].issame(t1)
    assert not tl[2].issame(t2)
    assert not tl[-1].issame(t2)

    # Create from a internal tensor.
    t = o3c.Tensor.ones((4, 2, 3), dtype, device)
    tl = o3c.TensorList.from_tensor(o3c.Tensor.ones((4, 2, 3), dtype, device))
    assert tl.element_shape == o3c.SizeVector([2, 3])
    assert tl.size == 4
    assert tl.dtype == dtype
    assert tl.device == device
    assert tl.is_resizable
    assert not tl.as_tensor().issame(t)

    # Create from a internal tensor, in-place.
    t = o3c.Tensor.ones((4, 2, 3), dtype, device)
    tl = o3c.TensorList.from_tensor(t, inplace=True)
    assert tl.element_shape == o3c.SizeVector([2, 3])
    assert tl.size == 4
    assert tl.dtype == dtype
    assert tl.device == device
    assert not tl.is_resizable
    assert tl.as_tensor().issame(t)


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_tensorlist_operation(device):
    dtype = o3c.Dtype.Float32
    t0 = o3c.Tensor.ones((2, 3), dtype, device) * 0
    t1 = o3c.Tensor.ones((2, 3), dtype, device) * 1
    t2 = o3c.Tensor.ones((2, 3), dtype, device) * 2

    # push_back
    tl = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    assert tl.size == 0
    tl.push_back(t0)
    assert tl.size == 1

    # resize
    tl.resize(10)
    assert tl.size == 10
    tl.resize(1)
    assert tl.size == 1

    # extend
    tl = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl.push_back(t0)
    tl_other = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl_other.push_back(t1)
    tl_other.push_back(t2)
    tl.extend(tl_other)
    assert tl.size == 3
    assert tl[0].allclose(t0)
    assert tl[1].allclose(t1)
    assert tl[2].allclose(t2)

    # +=
    tl = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl.push_back(t0)
    tl_other = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl_other.push_back(t1)
    tl_other.push_back(t2)
    tl += tl_other
    assert tl.size == 3
    assert tl[0].allclose(t0)
    assert tl[1].allclose(t1)
    assert tl[2].allclose(t2)

    # concat
    tl = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl.push_back(t0)
    tl_other = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl_other.push_back(t1)
    tl_combined = o3c.TensorList.concat(tl, tl_other)
    assert tl_combined.size == 2
    assert tl_combined[0].allclose(t0)
    assert tl_combined[1].allclose(t1)

    # +
    tl = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl.push_back(t0)
    tl_other = o3c.TensorList(o3c.SizeVector([2, 3]), dtype, device)
    tl_other.push_back(t1)
    tl_combined = tl + tl_other
    assert tl_combined.size == 2
    assert tl_combined[0].allclose(t0)
    assert tl_combined[1].allclose(t1)
