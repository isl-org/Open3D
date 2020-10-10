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

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_tensorlistmap(device):
    dtype = o3c.Dtype.Float32

    # Constructor.
    tlm = o3d.t.geometry.TensorListMap("points")

    # Get primary key().
    assert tlm.get_primary_key() == "points"

    # Map member access, assignment and "contains" check. This should be the
    # preferrred way to construct a TensorListMap with values in python.
    points = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    colors = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    tlm = o3d.t.geometry.TensorListMap("points")
    assert "points" not in tlm
    tlm["points"] = points
    assert "points" in tlm
    assert "colors" not in tlm
    tlm["colors"] = colors
    assert "colors" in tlm

    # Constructor with tl values.
    tlm = o3d.t.geometry.TensorListMap("points", {
        "points": points,
        "colors": colors
    })

    # Syncronized pushback.
    one_point = o3c.Tensor.ones((3,), dtype, device)
    one_color = o3c.Tensor.ones((3,), dtype, device)
    tlm.synchronized_push_back({"points": one_point, "colors": one_color})
    tlm["points"].push_back(one_point)
    assert not tlm.is_size_synchronized()
    with pytest.raises(RuntimeError):
        tlm.assert_size_synchronized()
    with pytest.raises(RuntimeError):
        tlm.synchronized_push_back({"points": one_point, "colors": one_color})
