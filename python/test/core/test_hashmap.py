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
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9, 9],
                             dtype=o3d.core.Dtype.Int64,
                             device=device)
    iterators, masks = hashmap.insert(keys, values)
    assert masks.to(o3d.core.Dtype.Int64).sum() == 5

    keys, values = hashmap.unpack_iterators(iterators, masks)

    assert keys[0] == 1
    assert keys[1] == 3
    assert keys[2] == 5
    assert keys[3] == 7

    assert values[0] == 100
    assert values[1] == 300
    assert values[2] == 500
    assert values[3] == 700

    # randomly in 4, 5
    assert masks[4:].to(o3d.core.Dtype.Int64).sum() == 1
    if masks[4]:
        assert keys[4] == 9
        assert values[4] == 900
    elif masks[5]:
        assert keys[5] == 9
        assert values[5] == 900


@pytest.mark.parametrize("device", list_devices())
def test_activate(device):
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    iterators, masks = hashmap.activate(keys)
    assert masks.to(o3d.core.Dtype.Int64).sum() == 5

    keys, _ = hashmap.unpack_iterators(iterators, masks)

    assert keys[0] == 1
    assert keys[1] == 3
    assert keys[2] == 5
    assert keys[3] == 7

    # randomly in 4, 5
    assert masks[4:].to(o3d.core.Dtype.Int64).sum() == 1
    if masks[4]:
        assert keys[4] == 9
    elif masks[5]:
        assert keys[5] == 9


@pytest.mark.parametrize("device", list_devices())
def test_find(device):
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
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
    iterators, masks = hashmap.find(keys)
    keys, values = hashmap.unpack_iterators(iterators, masks)

    assert masks[0].item() == True
    assert masks[1].item() == False
    assert masks[2].item() == True

    assert keys[0].item() == 100
    assert keys[2].item() == 500

    assert values[0].item() == 1
    assert values[2].item() == 5


@pytest.mark.parametrize("device", list_devices())
def test_erase(device):
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
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

    assert masks[0].item() == True
    assert masks[1].item() == False
    assert masks[2].item() == True


@pytest.mark.parametrize("device", list_devices())
def test_tensorwrapper(device):
    hashmap = o3d.core.Hashmap(10, o3d.core.Dtype.Int64, o3d.core.Dtype.Int64,
                               device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9],
                             dtype=o3d.core.Dtype.Int64,
                             device=device)
    hashmap.insert(keys, values)

    key_tensor = hashmap.get_key_blob_as_tensor((hashmap.capacity(),),
                                                o3d.core.Dtype.Int64)
    value_tensor = hashmap.get_value_blob_as_tensor((hashmap.capacity(),),
                                                    o3d.core.Dtype.Int64)

    keys = o3d.core.Tensor([100, 200, 500],
                           dtype=o3d.core.Dtype.Int64,
                           device=device)
    masks = hashmap.erase(keys)

    assert masks[0].item() == True
    assert masks[1].item() == False
    assert masks[2].item() == True
