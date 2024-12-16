# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest
import pickle
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


class WrongType():
    pass


@pytest.mark.parametrize("device", list_devices())
def test_tensormap(device):
    dtype = o3c.float32

    # Constructor.
    tm = o3d.t.geometry.TensorMap("positions")

    # Get primary key().
    assert tm.primary_key == "positions"

    # Map member access, assignment and "contains" check. This should be the
    # preferred way to construct a TensorMap with values in python.
    points = o3c.Tensor.ones((0, 3), dtype, device)
    colors = o3c.Tensor.ones((0, 3), dtype, device)
    tm = o3d.t.geometry.TensorMap("positions")
    assert "positions" not in tm
    tm.positions = points
    assert "positions" in tm
    assert "colors" not in tm
    tm.colors = colors
    assert "colors" in tm

    # Constructor with tm values.
    tm = o3d.t.geometry.TensorMap("positions", {
        "positions": points,
        "colors": colors
    })
    assert "positions" in tm
    assert "colors" in tm

    # __delitem__ operator.
    with pytest.raises(RuntimeError) as excinfo:
        del tm.positions
        assert 'cannot be deleted' in str(excinfo.value)

    # Test setter.
    tm = o3d.t.geometry.TensorMap("positions")

    # Set attributes.
    tm.positions = o3c.Tensor.ones((2, 3), dtype, device)
    tm.colors = o3c.Tensor.ones((2, 3), dtype, device)

    # Set attributes with numpy array.
    tm.positions = np.ones((3, 3), np.float32)
    tm.colors = np.ones((3, 3), np.float32)
    assert len(tm.positions) == 3
    assert len(tm.colors) == 3

    # Set existing attributes with wrong type.
    with pytest.raises(TypeError) as e:
        tm.positions = WrongType()

    # Set new attributes with wrong type.
    with pytest.raises(TypeError) as e:
        tm.normals = WrongType()

    # Set primary key.
    with pytest.raises(KeyError) as e:
        tm.primary_key = o3c.Tensor.ones((2, 3), dtype, device)

    # Test getter.
    tm = o3d.t.geometry.TensorMap("positions")
    assert isinstance(tm, o3d.t.geometry.TensorMap)

    # Set attributes.
    tm.positions = o3c.Tensor.ones((2, 3), dtype, device)
    tm.colors = o3c.Tensor.ones((2, 3), dtype, device)

    # Get existing attributes.
    colors = tm.colors
    assert len(colors) == 2

    # Get unknown attributes.
    with pytest.raises(KeyError) as e:
        normals = tm.normals

    # Get primary key.
    primary_key = tm.primary_key
    assert primary_key == "positions"


@pytest.mark.parametrize("device", list_devices())
def test_tensormap_modify(device):
    # Assigning to the *elements* of an alias will change the value in the map.
    # This tests that the alias shares the same memory as the tensor in the map.
    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    a_alias[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])

    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    tm.a[:] = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])

    # Assigning a new tensor to an alias should not change the tensor in the
    # map. The alias name simply points to a new tensor.
    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    a_alias = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [100])

    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    tm.a = o3c.Tensor([200], device=device)
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])

    # Pybind always returns a "shallow copy" of the tensor. This is a copy since
    # the new variable points to a different tensor object, and thus the id() is
    # different. The copy is shallow because the new tensor shares the same
    # memory as the tensor in the map.
    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    assert id(a_alias) != id(tm.a)

    # After deleting the key-value from the map, the alias shall still be alive.
    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    a_alias = tm.a
    assert len(tm) == 1
    del tm.a
    assert len(tm) == 0
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    a_alias[:] = 200
    np.testing.assert_equal(a_alias.cpu().numpy(), [200])

    # With this, swapping elements are supported.
    tm = o3d.t.geometry.TensorMap("positions")
    tm.a = o3c.Tensor([100], device=device)
    tm.b = o3c.Tensor([200], device=device)
    a_alias = tm.a
    b_alias = tm.b
    tm.a, tm.b = tm.b, tm.a
    np.testing.assert_equal(a_alias.cpu().numpy(), [100])
    np.testing.assert_equal(b_alias.cpu().numpy(), [200])
    np.testing.assert_equal(tm.a.cpu().numpy(), [200])
    np.testing.assert_equal(tm.b.cpu().numpy(), [100])


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


@pytest.mark.parametrize("device", list_devices())
def test_pickle(device):
    tm = o3d.t.geometry.TensorMap("positions")
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/tm.pkl"
        tm.positions = o3c.Tensor.ones((10, 3), o3c.float32, device=device)
        pickle.dump(tm, open(file_name, "wb"))
        tm_load = pickle.load(open(file_name, "rb"))
        assert tm_load.positions.device == device and tm_load.positions.dtype == o3c.float32
        np.testing.assert_equal(tm.positions.cpu().numpy(),
                                tm_load.positions.cpu().numpy())
