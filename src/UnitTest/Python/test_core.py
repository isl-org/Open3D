# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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
import torch
import torch.utils.dlpack
import time
import pytest


def list_devices():
    devices = [o3d.Device("CPU:" + str(0))]
    if (o3d.cuda.device_count() != torch.cuda.device_count()):
        raise RuntimeError(
            "o3d.cuda.device_count() != torch.cuda.device_count(), "
            f"{o3d.cuda.device_count()} != {torch.cuda.device_count()}")
    for i in range(o3d.cuda.device_count()):
        devices.append(o3d.Device("CUDA:" + str(i)))
    return devices


def test_dtype():
    dtype = o3d.Dtype.Int32
    assert o3d.DtypeUtil.byte_size(dtype) == 4
    assert o3d.DtypeUtil.to_string(dtype) == "Int32"


def test_device():
    device = o3d.Device()
    assert device.get_type() == o3d.Device.DeviceType.CPU
    assert device.get_id() == 0

    device = o3d.Device("CUDA", 1)
    assert device.get_type() == o3d.Device.DeviceType.CUDA
    assert device.get_id() == 1

    device = o3d.Device("CUDA:2")
    assert device.get_type() == o3d.Device.DeviceType.CUDA
    assert device.get_id() == 2

    assert o3d.Device("CUDA", 1) == o3d.Device("CUDA:1")
    assert o3d.Device("CUDA", 1) != o3d.Device("CUDA:0")

    assert o3d.Device("CUDA", 1).to_string() == "CUDA:1"


def test_size_vector():
    # List
    sv = o3d.SizeVector([-1, 2, 3])
    assert f"{sv}" == "{-1, 2, 3}"

    # Tuple
    sv = o3d.SizeVector((-1, 2, 3))
    assert f"{sv}" == "{-1, 2, 3}"

    # Numpy 1D array
    sv = o3d.SizeVector(np.array([-1, 2, 3]))
    assert f"{sv}" == "{-1, 2, 3}"

    # Empty
    sv = o3d.SizeVector()
    assert f"{sv}" == "{}"
    sv = o3d.SizeVector([])
    assert f"{sv}" == "{}"
    sv = o3d.SizeVector(())
    assert f"{sv}" == "{}"
    sv = o3d.SizeVector(np.array([]))
    assert f"{sv}" == "{}"

    # Automatic int casting (not rounding to nearest)
    sv = o3d.SizeVector((1.9, 2, 3))
    assert f"{sv}" == "{1, 2, 3}"

    # Automatic casting negative
    sv = o3d.SizeVector((-1.5, 2, 3))
    assert f"{sv}" == "{-1, 2, 3}"

    # 2D list exception
    with pytest.raises(ValueError):
        sv = o3d.SizeVector([[1, 2], [3, 4]])

    # 2D Numpy array exception
    with pytest.raises(ValueError):
        sv = o3d.SizeVector(np.array([[1, 2], [3, 4]]))

    # Garbage input
    with pytest.raises(ValueError):
        sv = o3d.SizeVector(["foo", "bar"])


def test_tensor_constructor():
    dtype = o3d.Dtype.Int32
    device = o3d.Device("CPU:0")

    # Numpy array
    np_t = np.array([[0, 1, 2], [3, 4, 5]])
    o3_t = o3d.Tensor(np_t, dtype, device)
    np.testing.assert_equal(np_t, o3_t.numpy())

    # 2D list
    li_t = [[0, 1, 2], [3, 4, 5]]
    o3_t = o3d.Tensor(li_t, dtype, device)
    np.testing.assert_equal(li_t, o3_t.numpy())

    # 2D list, inconsistent length
    li_t = [[0, 1, 2], [3, 4]]
    with pytest.raises(ValueError):
        o3_t = o3d.Tensor(li_t, dtype, device)

    # Automatic casting
    np_t_double = np.array([[0., 1.5, 2.], [3., 4., 5.]])
    np_t_int = np.array([[0, 1, 2], [3, 4, 5]])
    o3_t = o3d.Tensor(np_t_double, dtype, device)
    np.testing.assert_equal(np_t_int, o3_t.numpy())

    # Special strides
    np_t = np.random.randint(10, size=(10, 10))[1:10:2, 1:10:3].T
    o3_t = o3d.Tensor(np_t, dtype, device)
    np.testing.assert_equal(np_t, o3_t.numpy())


def test_tensor_from_to_numpy():
    # a->b copy; b, c share memory
    a = np.ones((2, 2))
    b = o3d.Tensor(a)
    c = b.numpy()

    c[0, 1] = 200
    r = np.array([[1., 200.], [1., 1.]])
    np.testing.assert_equal(r, b.numpy())
    np.testing.assert_equal(r, c)

    # a, b, c share memory
    a = np.array([[1., 1.], [1., 1.]])
    b = o3d.Tensor.from_numpy(a)
    c = b.numpy()

    a[0, 0] = 100
    c[0, 1] = 200
    r = np.array([[100., 200.], [1., 1.]])
    np.testing.assert_equal(r, a)
    np.testing.assert_equal(r, b.numpy())
    np.testing.assert_equal(r, c)

    # Special strides
    ran_t = np.random.randint(10, size=(10, 10)).astype(np.int32)
    src_t = ran_t[1:10:2, 1:10:3].T
    o3d_t = o3d.Tensor.from_numpy(src_t)  # Shared memory
    dst_t = o3d_t.numpy()
    np.testing.assert_equal(dst_t, src_t)

    dst_t[0, 0] = 100
    np.testing.assert_equal(dst_t, src_t)
    np.testing.assert_equal(dst_t, o3d_t.numpy())

    src_t[0, 1] = 200
    np.testing.assert_equal(dst_t, src_t)
    np.testing.assert_equal(dst_t, o3d_t.numpy())


def test_tensor_to_numpy_scope():
    src_t = np.array([[10., 11., 12.], [13., 14., 15.]])

    def get_dst_t():
        o3d_t = o3d.Tensor(src_t)  # Copy
        dst_t = o3d_t.numpy()
        return dst_t

    dst_t = get_dst_t()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("device", list_devices())
def test_tensor_to_pytorch_scope(device):
    src_t = np.array([[10, 11, 12.], [13., 14., 15.]])

    def get_dst_t():
        o3d_t = o3d.Tensor(src_t, device=device)  # Copy
        dst_t = torch.utils.dlpack.from_dlpack(o3d_t.to_dlpack())
        return dst_t

    dst_t = get_dst_t().cpu().numpy()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("device", list_devices())
def test_tensor_from_to_pytorch(device):
    device_id = device.get_id()
    device_type = device.get_type()

    # a, b, c share memory
    a = torch.ones((2, 2))
    if device_type == o3d.Device.DeviceType.CUDA:
        a = a.cuda(device_id)
    b = o3d.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(a))
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
    if device_type == o3d.Device.DeviceType.CUDA:
        th_t = th_t.cuda(device_id)

    o3_t = o3d.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_t))
    np.testing.assert_equal(th_t.cpu().numpy(), o3_t.cpu().numpy())

    th_t[0, 0] = 100
    np.testing.assert_equal(th_t.cpu().numpy(), o3_t.cpu().numpy())


def test_tensor_numpy_to_open3d_to_pytorch():
    # Numpy -> Open3D -> PyTorch all share the same memory
    a = np.ones((2, 2))
    b = o3d.Tensor.from_numpy(a)
    c = torch.utils.dlpack.from_dlpack(b.to_dlpack())

    a[0, 0] = 100
    c[0, 1] = 200
    r = np.array([[100., 200.], [1., 1.]])
    np.testing.assert_equal(r, a)
    np.testing.assert_equal(r, b.cpu().numpy())
    np.testing.assert_equal(r, c.cpu().numpy())
