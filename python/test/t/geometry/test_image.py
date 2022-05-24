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
def test_buffer_protocol_cpu(device):
    if device.get_type() == o3c.Device.DeviceType.CPU:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        dst_t = np.asarray(im)
        np.testing.assert_array_equal(src_t, dst_t)

        # Check that the memory is shared.
        dst_t[0, 0, 0] = 100
        new_dst_t = np.asarray(im)
        np.testing.assert_array_equal(dst_t, new_dst_t)
    else:
        # (rows, cols) -> (rows, cols, 1)
        src_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        # Ideally we shall test exception if .cpu() is not called, but
        # pytest.raises() cannot catch this exception for some reason.
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t[..., None], dst_t)

        # (rows, cols, channels) -> (rows, cols, channels)
        src_t = np.arange(18, dtype=np.float32).reshape((2, 3, 3))
        im = o3d.t.geometry.Image(o3d.core.Tensor.from_numpy(src_t))
        im = im.to(device=device)
        dst_t = np.asarray(im.cpu())
        np.testing.assert_array_equal(src_t, dst_t)
