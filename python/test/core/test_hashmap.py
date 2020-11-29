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

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_creation(device):
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    assert hashmap.size() == 0


@pytest.mark.parametrize("device", list_devices())
def test_insertion(device):
    capacity = 10
    hashmap = o3d.core.Hashmap(capacity, o3d.core.Dtype.Int64,
                               o3d.core.Dtype.Int64, device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9, 9],
                             dtype=o3d.core.Dtype.Int64,
                             device=device)
    addrs, masks = hashmap.insert(keys, values)
    assert masks.to(o3d.core.Dtype.Int64).sum() == 5

    valid_indices = addrs[masks].to(o3d.core.Dtype.Int64)
    valid_keys = hashmap.reinterpret_buffer_as_tensor(
        hashmap.get_key_tensor(), (capacity,),
        o3d.core.Dtype.Int64)[valid_indices]
    valid_values = hashmap.reinterpret_buffer_as_tensor(
        hashmap.get_value_tensor(), (capacity,),
        o3d.core.Dtype.Int64)[valid_indices]
    np.testing.assert_equal(valid_keys.cpu().numpy(),
                            valid_values.cpu().numpy() * 100)


@pytest.mark.parametrize("device", list_devices())
def test_activate(device):
    capacity = 10
    hashmap = o3d.core.Hashmap(capacity, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    addrs, masks = hashmap.activate(keys)
    assert masks.to(o3d.core.Dtype.Int64).sum() == 5

    valid_indices = addrs[masks].to(o3d.core.Dtype.Int64)
    valid_keys = hashmap.reinterpret_buffer_as_tensor(
        hashmap.get_key_tensor(), (capacity,),
        o3d.core.Dtype.Int64)[valid_indices]
    np.testing.assert_equal(np.sort(valid_keys.cpu().numpy()),
                            np.array([100, 300, 500, 700, 900]))


@pytest.mark.parametrize("device", list_devices())
def test_find(device):
    capacity = 10
    hashmap = o3d.core.Hashmap(capacity, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9],
                             dtype=o3d.core.Dtype.Int64,
                             device=device)
    hashmap.insert(keys, values)

    keys = o3d.core.Tensor([100, 200, 500],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    addrs, masks = hashmap.find(keys)
    np.testing.assert_equal(masks.cpu().numpy(),
                            np.array([True, False, True]))

    valid_indices = addrs[masks].to(o3d.core.Dtype.Int64)
    valid_keys = hashmap.reinterpret_buffer_as_tensor(
        hashmap.get_key_tensor(), (capacity,),
        o3d.core.Dtype.Int64)[valid_indices]
    valid_values = hashmap.reinterpret_buffer_as_tensor(
        hashmap.get_value_tensor(), (capacity,),
        o3d.core.Dtype.Int64)[valid_indices]
    assert valid_keys.shape[0] == 2

    np.testing.assert_equal(valid_keys.cpu().numpy(),
                            np.array([100, 500]))
    np.testing.assert_equal(valid_values.cpu().numpy(),
                            np.array([1, 5]))


@pytest.mark.parametrize("device", list_devices())
def test_erase(device):
    capacity = 10
    hashmap = o3d.core.Hashmap(capacity, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9],
                             dtype=o3d.core.Dtype.Int64,
                             device=device)
    hashmap.insert(keys, values)

    keys = o3d.core.Tensor([100, 200, 500],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    masks = hashmap.erase(keys)

    np.testing.assert_equal(masks.cpu().numpy(),
                            np.array([True, False, True]))

