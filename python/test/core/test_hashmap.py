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

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_creation(device):
    hashmap = o3c.HashMap(10, o3c.int64, [1], o3c.int64, [1], device)
    assert hashmap.size() == 0


@pytest.mark.parametrize("device", list_devices())
def test_insertion(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, [1], o3c.int64, [1], device)
    keys = o3c.Tensor([100, 300, 500, 700, 900, 900],
                      dtype=o3c.int64,
                      device=device)
    values = o3c.Tensor([1, 3, 5, 7, 9, 9], dtype=o3c.int64, device=device)
    buf_indices, masks = hashmap.insert(keys, values)
    assert masks.to(o3c.int64).sum() == 5

    valid_indices = buf_indices[masks].to(o3c.int64)
    valid_keys = hashmap.key_tensor()[valid_indices]
    valid_values = hashmap.value_tensor()[valid_indices]
    np.testing.assert_equal(valid_keys.cpu().numpy(),
                            valid_values.cpu().numpy() * 100)


@pytest.mark.parametrize("device", list_devices())
def test_activate(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, [1], o3c.int64, [1], device)
    keys = o3c.Tensor([100, 300, 500, 700, 900, 900],
                      dtype=o3c.int64,
                      device=device)
    buf_indices, masks = hashmap.activate(keys)
    assert masks.to(o3c.int64).sum() == 5

    valid_indices = buf_indices[masks].to(o3c.int64)
    valid_keys = hashmap.key_tensor()[valid_indices]
    np.testing.assert_equal(np.sort(valid_keys.cpu().numpy().flatten()),
                            np.array([100, 300, 500, 700, 900]))


@pytest.mark.parametrize("device", list_devices())
def test_find(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, [1], o3c.int64, [1], device)
    keys = o3c.Tensor([100, 300, 500, 700, 900], dtype=o3c.int64, device=device)
    values = o3c.Tensor([1, 3, 5, 7, 9], dtype=o3c.int64, device=device)
    hashmap.insert(keys, values)

    keys = o3c.Tensor([100, 200, 500], dtype=o3c.int64, device=device)
    buf_indices, masks = hashmap.find(keys)
    np.testing.assert_equal(masks.cpu().numpy(), np.array([True, False, True]))

    valid_indices = buf_indices[masks].to(o3c.int64)
    valid_keys = hashmap.key_tensor()[valid_indices]
    valid_values = hashmap.value_tensor()[valid_indices]
    assert valid_keys.shape[0] == 2

    np.testing.assert_equal(valid_keys.cpu().numpy().flatten(),
                            np.array([100, 500]))
    np.testing.assert_equal(valid_values.cpu().numpy().flatten(),
                            np.array([1, 5]))


@pytest.mark.parametrize("device", list_devices())
def test_erase(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, [1], o3c.int64, [1], device)
    keys = o3c.Tensor([100, 300, 500, 700, 900], dtype=o3c.int64, device=device)
    values = o3c.Tensor([1, 3, 5, 7, 9], dtype=o3c.int64, device=device)
    hashmap.insert(keys, values)

    keys = o3c.Tensor([100, 200, 500], dtype=o3c.int64, device=device)
    masks = hashmap.erase(keys)

    np.testing.assert_equal(masks.cpu().numpy(), np.array([True, False, True]))

    assert hashmap.size() == 3
    active_buf_indices = hashmap.active_buf_indices()
    active_indices = active_buf_indices.to(o3c.int64)

    active_keys = hashmap.key_tensor()[active_indices]
    active_values = hashmap.value_tensor()[active_indices]

    active_keys_np = active_keys.cpu().numpy().flatten()
    active_values_np = active_values.cpu().numpy().flatten()
    sorted_i = np.argsort(active_keys_np)
    np.testing.assert_equal(active_keys_np[sorted_i], np.array([300, 700, 900]))
    np.testing.assert_equal(active_values_np[sorted_i], np.array([3, 7, 9]))


@pytest.mark.parametrize("device", list_devices())
def test_complex_shape(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, [3], o3c.int64, [1], device)
    keys = o3c.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                      dtype=o3c.int64,
                      device=device)
    values = o3c.Tensor([1, 2, 3], dtype=o3c.int64, device=device)
    buf_indices, masks = hashmap.insert(keys, values)
    assert masks.to(o3c.int64).sum() == 3

    valid_indices = buf_indices[masks].to(o3c.int64)
    valid_keys = hashmap.key_tensor()[valid_indices, :]

    valid_values = hashmap.value_tensor()[valid_indices]

    np.testing.assert_equal(
        valid_keys.cpu().numpy().flatten(),
        np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]).flatten())
    np.testing.assert_equal(valid_values.cpu().numpy().flatten(),
                            np.array([1, 2, 3]))

    keys = o3c.Tensor([[1, 2, 3], [4, 5, 6]], dtype=o3c.int64, device=device)
    masks = hashmap.erase(keys)

    np.testing.assert_equal(masks.cpu().numpy().flatten(),
                            np.array([True, False]))


@pytest.mark.parametrize("device", list_devices())
def test_multivalue(device):
    capacity = 10
    hashmap = o3c.HashMap(capacity, o3c.int64, (3), (o3c.int64, o3c.float64),
                          ((1), (1)), device)
    keys = o3c.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                      dtype=o3c.int64,
                      device=device)
    values_i64 = o3c.Tensor([1, 2, 3], dtype=o3c.int64, device=device)
    values_f64 = o3c.Tensor([400.0, 500.0, 600.0],
                            dtype=o3c.float64,
                            device=device)
    buf_indices, masks = hashmap.insert(keys, [values_i64, values_f64])
    assert masks.to(o3c.int64).sum() == 3

    valid_indices = buf_indices[masks].to(o3c.int64)
    valid_keys = hashmap.key_tensor()[valid_indices, :]

    valid_values_i64 = hashmap.value_tensor(0)[valid_indices]
    valid_values_f64 = hashmap.value_tensor(1)[valid_indices]

    np.testing.assert_equal(
        valid_keys.cpu().numpy().flatten(),
        np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]).flatten())
    np.testing.assert_equal(valid_values_i64.cpu().numpy().flatten(),
                            np.array([1, 2, 3]))
    np.testing.assert_allclose(valid_values_f64.cpu().numpy().flatten(),
                               np.array([400.0, 500.0, 600.0]))

    keys = o3c.Tensor([[1, 2, 3], [4, 5, 6]], dtype=o3c.int64, device=device)
    masks = hashmap.erase(keys)

    np.testing.assert_equal(masks.cpu().numpy().flatten(),
                            np.array([True, False]))


@pytest.mark.parametrize("device", list_devices())
def test_hashset(device):
    capacity = 10
    hashset = o3c.HashSet(capacity, o3c.int64, (3), device)
    keys = o3c.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                      dtype=o3c.int64,
                      device=device)
    buf_indices, masks = hashset.insert(keys)
    assert masks.to(o3c.int64).sum() == 3

    keys = o3c.Tensor([[1, 2, 3], [3, 4, 5], [4, 5, 6]],
                      dtype=o3c.int64,
                      device=device)
    buf_indices, masks = hashset.find(keys)
    np.testing.assert_equal(masks.cpu().numpy().flatten(),
                            np.array([True, True, False]))

    keys = o3c.Tensor([[1, 2, 3], [4, 5, 6]], dtype=o3c.int64, device=device)
    masks = hashset.erase(keys)

    np.testing.assert_equal(masks.cpu().numpy().flatten(),
                            np.array([True, False]))
