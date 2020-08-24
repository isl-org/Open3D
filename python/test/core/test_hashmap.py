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
import core_test_utils


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_creation(device):
    hashmap = o3d.core.Hashmap(10, 8, 8, device)
    print(hashmap.size())


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_insertion(device):
    hashmap = o3d.core.Hashmap(10, 8, 8, device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900], device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9], device=device)
    hashmap.insert(keys, values)


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_find(device):
    hashmap = o3d.core.Hashmap(10, 8, 8, device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900], device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9], device=device)
    hashmap.insert(keys, values)

    keys = o3d.core.Tensor([100, 200, 500], device=device)
    masks_c = hashmap.find(keys)
    masks = o3d.core.Tensor([])
    masks.shallow_copy_from(masks_c)

    assert masks[0].item() == True
    assert masks[1].item() == False
    assert masks[2].item() == True


@pytest.mark.parametrize("device", core_test_utils.list_devices())
def test_erase(device):
    hashmap = o3d.core.Hashmap(10, 8, 8, device)
    keys = o3d.core.Tensor([100, 300, 500, 700, 900], device=device)
    values = o3d.core.Tensor([1, 3, 5, 7, 9], device=device)
    hashmap.insert(keys, values)

    keys = o3d.core.Tensor([100, 200, 500], device=device)
    masks_c = hashmap.erase(keys)
    masks = o3d.core.Tensor([])
    masks.shallow_copy_from(masks_c)

    assert masks[0].item() == True
    assert masks[1].item() == False
    assert masks[2].item() == True
