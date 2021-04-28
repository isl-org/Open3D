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
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_tensormap(device):
    dtype = o3c.Dtype.Float32

    # Constructor.
    tl = o3d.t.geometry.TensorMap("points")

    # Get primary key().
    assert tl.get_primary_key() == "points"

    # Map member access, assignment and "contains" check. This should be the
    # preferrred way to construct a TensorMap with values in python.
    points = o3c.Tensor.ones((0, 3), dtype, device)
    colors = o3c.Tensor.ones((0, 3), dtype, device)
    tl = o3d.t.geometry.TensorMap("points")
    assert "points" not in tl
    tl["points"] = points
    assert "points" in tl
    assert "colors" not in tl
    tl["colors"] = colors
    assert "colors" in tl

    # Constructor with tl values.
    tl = o3d.t.geometry.TensorMap("points", {
        "points": points,
        "colors": colors
    })
    assert "points" in tl
    assert "colors" in tl

    # __delitem__ operator.
    with pytest.raises(RuntimeError) as excinfo:
        del tl["points"]
        assert 'cannot be deleted' in str(excinfo.value)
