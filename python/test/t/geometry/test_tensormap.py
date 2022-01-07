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

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_tensormap(device):
    dtype = o3c.float32

    # Constructor.
    tl = o3d.t.geometry.TensorMap("positions")

    # Get primary key().
    assert tl.get_primary_key() == "positions"

    # Map member access, assignment and "contains" check. This should be the
    # preferrred way to construct a TensorMap with values in python.
    points = o3c.Tensor.ones((0, 3), dtype, device)
    colors = o3c.Tensor.ones((0, 3), dtype, device)
    tl = o3d.t.geometry.TensorMap("positions")
    assert "positions" not in tl
    tl["positions"] = points
    assert "positions" in tl
    assert "colors" not in tl
    tl["colors"] = colors
    assert "colors" in tl

    # Constructor with tl values.
    tl = o3d.t.geometry.TensorMap("positions", {
        "positions": points,
        "colors": colors
    })
    assert "positions" in tl
    assert "colors" in tl

    # __delitem__ operator.
    with pytest.raises(RuntimeError) as excinfo:
        del tl["positions"]
        assert 'cannot be deleted' in str(excinfo.value)


@pytest.mark.parametrize("device", list_devices())
def test_tensormap_modify(device):
    # Assigning to the *elements* of an alias will change the value in the map.
    # This tests that the alias shares the same memory as the tensor in the map.
    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    a_alias[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    tm["a"][:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    # Assigning a new tensor to an alias should not change the tensor in the
    # map. The alias name simply points to a new tensor.
    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    a_alias = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [100])

    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    tm["a"] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    # Pybind always returns a "shallow copy" of the tensor. This is a copy since
    # the new variable points to a different tensor object, and thus the id() is
    # different. The copy is shallow because the new tensor shares the same
    # memory as the tensor in the map.
    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    assert id(a_alias) != id(tm["a"])

    # After deleting the key-value from the map, the alias shall still be alive.
    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    assert len(tm) == 1
    del tm["a"]
    assert len(tm) == 0
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])

    # With this, swapping elements are supported.
    tm = o3d.t.geometry.TensorMap("positions")
    tm["a"] = o3c.Tensor([100], device=device)
    tm["b"] = o3c.Tensor([200], device=device)
    a_alias = tm["a"]
    b_alias = tm["b"]
    tm["a"], tm["b"] = tm["b"], tm["a"]
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(b_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])
    np.testing.assert_equal(tm["b"].cpu().numpy(), [100])


@pytest.mark.parametrize("device", list_devices())
def test_tensor_dict_modify(device):
    """
    Same as test_tensormap_modify(), but we put Tensors in a python dict.
    The only difference is that the id of the alias will be the same.
    """
    # Assign to elements.
    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    a_alias[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    tm["a"][:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    # Assign a new tensor.
    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    a_alias = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [100])

    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    tm["a"] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])

    # Object id.
    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    assert id(a_alias) == id(tm["a"])

    # Liveness of alias.
    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    a_alias = tm["a"]
    assert len(tm) == 1
    del tm["a"]
    assert len(tm) == 0
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])

    # Swap.
    tm = dict()
    tm["a"] = o3c.Tensor([100], device=device)
    tm["b"] = o3c.Tensor([200], device=device)
    a_alias = tm["a"]
    b_alias = tm["b"]
    tm["a"], tm["b"] = tm["b"], tm["a"]
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(b_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm["a"].cpu().numpy(), [200])
    np.testing.assert_equal(tm["b"].cpu().numpy(), [100])


def test_numpy_dict_modify():
    """
    Same as test_tensor_dict_modify(), but we put numpy arrays in a python dict.
    The id of the alias will be the same.
    """
    # Assign to elements.
    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    a_alias[:] = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm["a"], [200])

    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    tm["a"][:] = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm["a"], [200])

    # Assign a new tensor.
    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    tm["a"] = np.array([200])
    np.testing.assert_equal(a_alias, [100])
    np.testing.assert_equal(tm["a"], [200])

    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    a_alias = np.array([200])
    np.testing.assert_equal(a_alias, [200])
    np.testing.assert_equal(tm["a"], [100])

    # Object id.
    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    assert id(a_alias) == id(tm["a"])

    # Liveness of alias.
    tm = dict()
    tm["a"] = np.array([100])
    a_alias = tm["a"]
    assert len(tm) == 1
    del tm["a"]
    assert len(tm) == 0
    np.testing.assert_equal(a_alias, [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias, [200])

    # Swap.
    tm = dict()
    tm["a"] = np.array([100])
    tm["b"] = np.array([200])
    a_alias = tm["a"]
    b_alias = tm["b"]
    tm["a"], tm["b"] = tm["b"], tm["a"]
    np.testing.assert_equal(a_alias, [100])
    np.testing.assert_equal(b_alias, [200])
    np.testing.assert_equal(tm["a"], [200])
    np.testing.assert_equal(tm["b"], [100])
