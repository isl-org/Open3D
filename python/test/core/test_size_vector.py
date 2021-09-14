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

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


def test_size_vector():
    # List
    sv = o3d.core.SizeVector([-1, 2, 3])
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Tuple
    sv = o3d.core.SizeVector((-1, 2, 3))
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Numpy 1D array
    sv = o3d.core.SizeVector(np.array([-1, 2, 3]))
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Empty
    sv = o3d.core.SizeVector()
    assert "{}".format(sv) == "SizeVector[]"
    sv = o3d.core.SizeVector([])
    assert "{}".format(sv) == "SizeVector[]"
    sv = o3d.core.SizeVector(())
    assert "{}".format(sv) == "SizeVector[]"
    sv = o3d.core.SizeVector(np.array([]))
    assert "{}".format(sv) == "SizeVector[]"

    # 1-dimensional SizeVector
    assert o3d.core.SizeVector(3) == (3,)
    assert o3d.core.SizeVector((3)) == (3,)
    assert o3d.core.SizeVector((3,)) == (3,)
    assert o3d.core.SizeVector([3]) == (3,)
    assert o3d.core.SizeVector([
        3,
    ]) == (3,)

    # Not integer: thorws exception
    with pytest.raises(Exception):
        sv = o3d.core.SizeVector([1.9, 2, 3])

    with pytest.raises(Exception):
        sv = o3d.core.SizeVector([-1.5, 2, 3])

    # 2D list exception
    with pytest.raises(Exception):
        sv = o3d.core.SizeVector([[1, 2], [3, 4]])

    # 2D Numpy array exception
    with pytest.raises(Exception):
        sv = o3d.core.SizeVector(np.array([[1, 2], [3, 4]]))

    # Garbage input
    with pytest.raises(Exception):
        sv = o3d.core.SizeVector(["foo", "bar"])


@pytest.mark.parametrize("device", list_devices())
def test_implicit_conversion(device):
    # Reshape
    t = o3d.core.Tensor.ones((3, 4), device=device)
    assert t.reshape(o3d.core.SizeVector((4, 3))).shape == (4, 3)
    assert t.reshape(o3d.core.SizeVector([4, 3])).shape == (4, 3)
    assert t.reshape((4, 3)).shape == (4, 3)
    assert t.reshape([4, 3]).shape == (4, 3)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.reshape((4, 3.0))
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.reshape((4.0, 3.0))
    with pytest.raises(RuntimeError, match="Invalid shape dimension"):
        t.reshape((4, -3))

    # 0-dimensional
    assert o3d.core.Tensor.ones((), device=device).shape == ()
    assert o3d.core.Tensor.ones([], device=device).shape == ()

    # 1-dimensional
    assert o3d.core.Tensor.ones(3, device=device).shape == (3,)
    assert o3d.core.Tensor.ones((3), device=device).shape == (3,)
    assert o3d.core.Tensor.ones((3,), device=device).shape == (3,)
    assert o3d.core.Tensor.ones([3], device=device).shape == (3,)
    assert o3d.core.Tensor.ones([
        3,
    ], device=device).shape == (3,)

    # Tensor creation
    assert o3d.core.Tensor.empty((3, 4), device=device).shape == (3, 4)
    assert o3d.core.Tensor.ones((3, 4), device=device).shape == (3, 4)
    assert o3d.core.Tensor.zeros((3, 4), device=device).shape == (3, 4)
    assert o3d.core.Tensor.full((3, 4), 10, device=device).shape == (3, 4)

    # Reduction
    t = o3d.core.Tensor.ones((3, 4, 5), device=device)
    assert t.sum(o3d.core.SizeVector([0, 2])).shape == (4,)
    assert t.sum(o3d.core.SizeVector([0, 2]), keepdim=True).shape == (1, 4, 1)
    assert t.sum((0, 2)).shape == (4,)
    assert t.sum([0, 2]).shape == (4,)
    assert t.sum((0, 2), keepdim=True).shape == (1, 4, 1)
    assert t.sum([0, 2], keepdim=True).shape == (1, 4, 1)
