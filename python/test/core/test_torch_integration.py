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

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices_with_torch, torch_available

if torch_available():
    import torch
    import torch.utils.dlpack


@pytest.mark.parametrize("device", list_devices_with_torch())
def test_tensor_to_pytorch_scope(device):
    if not torch_available():
        return

    src_t = np.array([[10, 11, 12.], [13., 14., 15.]])

    def get_dst_t():
        o3d_t = o3d.core.Tensor(src_t, device=device)  # Copy
        dst_t = torch.utils.dlpack.from_dlpack(o3d_t.to_dlpack())
        return dst_t

    dst_t = get_dst_t().cpu().numpy()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("device", list_devices_with_torch())
def test_tensor_from_to_pytorch(device):
    if not torch_available():
        return

    device_id = device.get_id()
    device_type = device.get_type()

    # a, b, c share memory
    a = torch.ones((2, 2))
    if device_type == o3d.core.Device.DeviceType.CUDA:
        a = a.cuda(device_id)
    b = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(a))
    c = torch.utils.dlpack.from_dlpack(b.to_dlpack())

    a[0, 0] = 100
    c[0, 1] = 200
    r = np.array([[100., 200.], [1., 1.]])
    np.testing.assert_equal(r, a.cpu().numpy())
    np.testing.assert_equal(r, b.cpu().numpy())
    np.testing.assert_equal(r, c.cpu().numpy())

    # Special strides
    np_r = np.random.randint(10, size=(10, 10)).astype(np.int32)
    th_r = torch.Tensor(np_r)
    th_t = th_r[1:10:2, 1:10:3].T
    if device_type == o3d.core.Device.DeviceType.CUDA:
        th_t = th_t.cuda(device_id)

    o3_t = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_t))
    np.testing.assert_equal(th_t.cpu().numpy(), o3_t.cpu().numpy())

    th_t[0, 0] = 100
    np.testing.assert_equal(th_t.cpu().numpy(), o3_t.cpu().numpy())


def test_tensor_numpy_to_open3d_to_pytorch():
    if not torch_available():
        return

    # Numpy -> Open3D -> PyTorch all share the same memory
    a = np.ones((2, 2))
    b = o3d.core.Tensor.from_numpy(a)
    c = torch.utils.dlpack.from_dlpack(b.to_dlpack())

    a[0, 0] = 100
    c[0, 1] = 200
    r = np.array([[100., 200.], [1., 1.]])
    np.testing.assert_equal(r, a)
    np.testing.assert_equal(r, b.cpu().numpy())
    np.testing.assert_equal(r, c.cpu().numpy())
