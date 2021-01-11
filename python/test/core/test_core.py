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
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_creation(device):
    # Shape takes tuple, list or o3d.core.SizeVector
    t = o3d.core.Tensor.empty((2, 3), o3d.core.Dtype.Float32, device=device)
    assert t.shape == o3d.core.SizeVector([2, 3])
    t = o3d.core.Tensor.empty([2, 3], o3d.core.Dtype.Float32, device=device)
    assert t.shape == o3d.core.SizeVector([2, 3])
    t = o3d.core.Tensor.empty(o3d.core.SizeVector([2, 3]),
                              o3d.core.Dtype.Float32,
                              device=device)
    assert t.shape == o3d.core.SizeVector([2, 3])

    # Test zeros and ones
    t = o3d.core.Tensor.zeros((2, 3), o3d.core.Dtype.Float32, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.zeros((2, 3), dtype=np.float32))
    t = o3d.core.Tensor.ones((2, 3), o3d.core.Dtype.Float32, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.ones((2, 3), dtype=np.float32))

    # Automatic casting of dtype.
    t = o3d.core.Tensor.full((2,), False, o3d.core.Dtype.Float32, device=device)
    np.testing.assert_equal(t.cpu().numpy(),
                            np.full((2,), False, dtype=np.float32))
    t = o3d.core.Tensor.full((2,), 3.5, o3d.core.Dtype.UInt8, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.full((2,), 3.5, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(), (0,), (1,), (0, 2), (0, 0, 2),
                                   (2, 0, 3)])
@pytest.mark.parametrize("device", list_devices())
def test_creation_special_shapes(shape, device):
    o3_t = o3d.core.Tensor.full(shape,
                                3.14,
                                o3d.core.Dtype.Float32,
                                device=device)
    np_t = np.full(shape, 3.14, dtype=np.float32)
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)


def test_dtype():
    dtype = o3d.core.Dtype.Int32
    assert dtype.byte_size() == 4
    assert "{}".format(dtype) == "Int32"


def test_device():
    device = o3d.core.Device()
    assert device.get_type() == o3d.core.Device.DeviceType.CPU
    assert device.get_id() == 0

    device = o3d.core.Device("CUDA", 1)
    assert device.get_type() == o3d.core.Device.DeviceType.CUDA
    assert device.get_id() == 1

    device = o3d.core.Device("CUDA:2")
    assert device.get_type() == o3d.core.Device.DeviceType.CUDA
    assert device.get_id() == 2

    assert o3d.core.Device("CUDA", 1) == o3d.core.Device("CUDA:1")
    assert o3d.core.Device("CUDA", 1) != o3d.core.Device("CUDA:0")

    assert o3d.core.Device("CUDA", 1).__str__() == "CUDA:1"


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

    # Not integer: thorws exception
    with pytest.raises(RuntimeError):
        sv = o3d.core.SizeVector([1.9, 2, 3])

    with pytest.raises(RuntimeError):
        sv = o3d.core.SizeVector([-1.5, 2, 3])

    # 2D list exception
    with pytest.raises(RuntimeError):
        sv = o3d.core.SizeVector([[1, 2], [3, 4]])

    # 2D Numpy array exception
    with pytest.raises(RuntimeError):
        sv = o3d.core.SizeVector(np.array([[1, 2], [3, 4]]))

    # Garbage input
    with pytest.raises(RuntimeError):
        sv = o3d.core.SizeVector(["foo", "bar"])


@pytest.mark.parametrize("device", list_devices())
def test_tensor_constructor(device):
    dtype = o3d.core.Dtype.Int32

    # Numpy array
    np_t = np.array([[0, 1, 2], [3, 4, 5]])
    o3_t = o3d.core.Tensor(np_t, dtype, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # 2D list
    li_t = [[0, 1, 2], [3, 4, 5]]
    no3_t = o3d.core.Tensor(li_t, dtype, device)
    np.testing.assert_equal(li_t, o3_t.cpu().numpy())

    # 2D list, inconsistent length
    li_t = [[0, 1, 2], [3, 4]]
    with pytest.raises(Exception):
        o3_t = o3d.core.Tensor(li_t, dtype, device)

    # Automatic casting
    np_t_double = np.array([[0., 1.5, 2.], [3., 4., 5.]])
    np_t_int = np.array([[0, 1, 2], [3, 4, 5]])
    o3_t = o3d.core.Tensor(np_t_double, dtype, device)
    np.testing.assert_equal(np_t_int, o3_t.cpu().numpy())

    # Special strides
    np_t = np.random.randint(10, size=(10, 10))[1:10:2, 1:10:3].T
    o3_t = o3d.core.Tensor(np_t, dtype, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Boolean
    np_t = np.array([True, False, True], dtype=np.bool)
    o3_t = o3d.core.Tensor([True, False, True], o3d.core.Dtype.Bool, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())
    o3_t = o3d.core.Tensor(np_t, o3d.core.Dtype.Bool, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_arange(device):
    # Full parameters.
    setups = [(0, 10, 1), (0, 10, 1), (0.0, 10.0, 2.0), (0.0, -10.0, -2.0)]
    for start, stop, step in setups:
        np_t = np.arange(start, stop, step)
        o3_t = o3d.core.Tensor.arange(start,
                                      stop,
                                      step,
                                      dtype=None,
                                      device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Only stop.
    for stop in [1.0, 2.0, 3.0, 1, 2, 3]:
        np_t = np.arange(stop)
        o3_t = o3d.core.Tensor.arange(stop, dtype=None, device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Only start, stop (step = 1).
    setups = [(0, 10), (0, 10), (0.0, 10.0), (0.0, -10.0)]
    for start, stop in setups:
        np_t = np.arange(start, stop)
        # Not full parameter list, need to specify device by kw.
        o3_t = o3d.core.Tensor.arange(start, stop, dtype=None, device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: int -> int.
    o3_t = o3d.core.Tensor.arange(0, 5, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3d.core.Dtype.Int64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: int, float -> float.
    o3_t = o3d.core.Tensor.arange(0, 5.0, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3d.core.Dtype.Float64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: float, float -> float.
    o3_t = o3d.core.Tensor.arange(0.0, 5.0, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3d.core.Dtype.Float64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: explicit type.
    o3_t = o3d.core.Tensor.arange(0.0,
                                  5.0,
                                  dtype=o3d.core.Dtype.Int64,
                                  device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3d.core.Dtype.Int64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())


def test_tensor_from_to_numpy():
    # a->b copy; b, c share memory
    a = np.ones((2, 2))
    b = o3d.core.Tensor(a)
    c = b.numpy()

    c[0, 1] = 200
    r = np.array([[1., 200.], [1., 1.]])
    np.testing.assert_equal(r, b.numpy())
    np.testing.assert_equal(r, c)

    # a, b, c share memory
    a = np.array([[1., 1.], [1., 1.]])
    b = o3d.core.Tensor.from_numpy(a)
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
    o3d_t = o3d.core.Tensor.from_numpy(src_t)  # Shared memory
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
        o3d_t = o3d.core.Tensor(src_t)  # Copy
        dst_t = o3d_t.numpy()
        return dst_t

    dst_t = get_dst_t()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("device", list_devices())
def test_binary_ew_ops(device):
    a = o3d.core.Tensor(np.array([4, 6, 8, 10, 12, 14]), device=device)
    b = o3d.core.Tensor(np.array([2, 3, 4, 5, 6, 7]), device=device)
    np.testing.assert_equal((a + b).cpu().numpy(),
                            np.array([6, 9, 12, 15, 18, 21]))
    np.testing.assert_equal((a - b).cpu().numpy(), np.array([2, 3, 4, 5, 6, 7]))
    np.testing.assert_equal((a * b).cpu().numpy(),
                            np.array([8, 18, 32, 50, 72, 98]))
    np.testing.assert_equal((a / b).cpu().numpy(), np.array([2, 2, 2, 2, 2, 2]))

    a = o3d.core.Tensor(np.array([4, 6, 8, 10, 12, 14]), device=device)
    a += b
    np.testing.assert_equal(a.cpu().numpy(), np.array([6, 9, 12, 15, 18, 21]))

    a = o3d.core.Tensor(np.array([4, 6, 8, 10, 12, 14]), device=device)
    a -= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([2, 3, 4, 5, 6, 7]))

    a = o3d.core.Tensor(np.array([4, 6, 8, 10, 12, 14]), device=device)
    a *= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([8, 18, 32, 50, 72, 98]))

    a = o3d.core.Tensor(np.array([4, 6, 8, 10, 12, 14]), device=device)
    a //= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([2, 2, 2, 2, 2, 2]))


@pytest.mark.parametrize("device", list_devices())
def test_to(device):
    a = o3d.core.Tensor(np.array([0.1, 1.2, 2.3, 3.4, 4.5,
                                  5.6]).astype(np.float32),
                        device=device)
    b = a.to(o3d.core.Dtype.Int32)
    np.testing.assert_equal(b.cpu().numpy(), np.array([0, 1, 2, 3, 4, 5]))
    assert b.shape == o3d.core.SizeVector([6])
    assert b.strides == o3d.core.SizeVector([1])
    assert b.dtype == o3d.core.Dtype.Int32
    assert b.device == a.device


@pytest.mark.parametrize("device", list_devices())
def test_unary_ew_ops(device):
    src_vals = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    src = o3d.core.Tensor(src_vals, device=device)

    rtol = 1e-5
    atol = 0
    np.testing.assert_allclose(src.sqrt().cpu().numpy(),
                               np.sqrt(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.sin().cpu().numpy(),
                               np.sin(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.cos().cpu().numpy(),
                               np.cos(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.neg().cpu().numpy(),
                               -src_vals,
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.exp().cpu().numpy(),
                               np.exp(src_vals),
                               rtol=rtol,
                               atol=atol)


@pytest.mark.parametrize("device", list_devices())
def test_getitem(device):
    np_t = np.array(range(24)).reshape((2, 3, 4))
    o3_t = o3d.core.Tensor(np_t, device=device)

    np.testing.assert_equal(o3_t[:].cpu().numpy(), np_t[:])
    np.testing.assert_equal(o3_t[0].cpu().numpy(), np_t[0])
    np.testing.assert_equal(o3_t[0, 1].cpu().numpy(), np_t[0, 1])
    np.testing.assert_equal(o3_t[0, :].cpu().numpy(), np_t[0, :])
    np.testing.assert_equal(o3_t[0, 1:3].cpu().numpy(), np_t[0, 1:3])
    np.testing.assert_equal(o3_t[0, :, :-2].cpu().numpy(), np_t[0, :, :-2])
    np.testing.assert_equal(o3_t[0, 1:3, 2].cpu().numpy(), np_t[0, 1:3, 2])
    np.testing.assert_equal(o3_t[0, 1:-1, 2].cpu().numpy(), np_t[0, 1:-1, 2])
    np.testing.assert_equal(o3_t[0, 1:3, 0:4:2].cpu().numpy(), np_t[0, 1:3,
                                                                    0:4:2])
    np.testing.assert_equal(o3_t[0, 1:3, 0:-1:2].cpu().numpy(), np_t[0, 1:3,
                                                                     0:-1:2])
    np.testing.assert_equal(o3_t[0, 1, :].cpu().numpy(), np_t[0, 1, :])

    # Slice out-of-range
    np.testing.assert_equal(o3_t[1:6].cpu().numpy(), np_t[1:6])
    np.testing.assert_equal(o3_t[2:5, -10:20].cpu().numpy(), np_t[2:5, -10:20])
    np.testing.assert_equal(o3_t[2:2, 3:3, 4:4].cpu().numpy(), np_t[2:2, 3:3,
                                                                    4:4])
    np.testing.assert_equal(o3_t[2:20, 3:30, 4:40].cpu().numpy(),
                            np_t[2:20, 3:30, 4:40])
    np.testing.assert_equal(o3_t[-2:20, -3:30, -4:40].cpu().numpy(),
                            np_t[-2:20, -3:30, -4:40])

    # Slice the slice
    np.testing.assert_equal(o3_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3].cpu().numpy(),
                            np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3])


@pytest.mark.parametrize("device", list_devices())
def test_setitem(device):
    np_ref = np.array(range(24)).reshape((2, 3, 4))
    o3_ref = o3d.core.Tensor(np_ref, device=device)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[:].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[:] = np_fill_t
    o3_t[:] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0] = np_fill_t
    o3_t[0] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1] = np_fill_t
    o3_t[0, 1] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, :].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, :] = np_fill_t
    o3_t[0, :] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1:3] = np_fill_t
    o3_t[0, 1:3] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, :, :-2].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, :, :-2] = np_fill_t
    o3_t[0, :, :-2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 2].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 2] = np_fill_t
    o3_t[0, 1:3, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:-1, 2].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1:-1, 2] = np_fill_t
    o3_t[0, 1:-1, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:4:2].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 0:4:2] = np_fill_t
    o3_t[0, 1:3, 0:4:2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:-1:2].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 0:-1:2] = np_fill_t
    o3_t[0, 1:3, 0:-1:2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1, :].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0, 1, :] = np_fill_t
    o3_t[0, 1, :] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.core.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3].shape)
    o3_fill_t = o3d.core.Tensor(np_fill_t, device=device)
    np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = np_fill_t
    o3_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_sum(dim, keepdim, device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.sum(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.sum(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("shape_and_axis", [
    ((), ()),
    ((0,), ()),
    ((0,), (0)),
    ((0, 2), ()),
    ((0, 2), (0)),
    ((0, 2), (1)),
])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_special_shapes(shape_and_axis, keepdim, device):
    shape, axis = shape_and_axis
    np_src = np.array(np.random.rand(*shape))
    o3_src = o3d.core.Tensor(np_src, device=device)
    np.testing.assert_equal(o3_src.cpu().numpy(), np_src)

    np_dst = np_src.sum(axis=axis, keepdims=keepdim)
    o3_dst = o3_src.sum(dim=axis, keepdim=keepdim)
    np.testing.assert_equal(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_mean(dim, keepdim, device):
    np_src = np.array(range(24)).reshape((2, 3, 4)).astype(np.float32)
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.mean(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.mean(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_prod(dim, keepdim, device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.prod(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.prod(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_min(dim, keepdim, device):
    np_src = np.array(range(24))
    np.random.shuffle(np_src)
    np_src = np_src.reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.min(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.min(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_max(dim, keepdim, device):
    np_src = np.array(range(24))
    np.random.shuffle(np_src)
    np_src = np_src.reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.max(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.max(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("dim", [0, 1, 2, None])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_argmin_argmax(dim, device):
    np_src = np.array(range(24))
    np.random.shuffle(np_src)
    np_src = np_src.reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src.argmin(axis=dim)
    o3_dst = o3_src.argmin(dim=dim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)

    np_dst = np_src.argmax(axis=dim)
    o3_dst = o3_src.argmax(dim=dim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("device", list_devices())
def test_advanced_index_get_mixed(device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_dst = np_src[1, 0:2, [1, 2]]
    o3_dst = o3_src[1, 0:2, [1, 2]]
    np.testing.assert_equal(o3_dst.cpu().numpy(), np_dst)

    # Subtle differences between slice and list
    np_src = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800]).reshape(3, 3)
    o3_src = o3d.core.Tensor(np_src, device=device)
    np.testing.assert_equal(o3_src[1, 2].cpu().numpy(), np_src[1, 2])
    np.testing.assert_equal(o3_src[[1, 2]].cpu().numpy(), np_src[[1, 2]])
    np.testing.assert_equal(o3_src[(1, 2)].cpu().numpy(), np_src[(1, 2)])
    np.testing.assert_equal(o3_src[(1, 2), [1, 2]].cpu().numpy(),
                            np_src[(1, 2), [1, 2]])

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3d.core.Tensor(np_src, device=device)
    o3_dst = o3_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np_dst = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np.testing.assert_equal(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("device", list_devices())
def test_advanced_index_set_mixed(device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.core.Tensor(np_src, device=device)

    np_fill = np.array(([[100, 200], [300, 400]]))
    o3_fill = o3d.core.Tensor(np_fill, device=device)

    np_src[1, 0:2, [1, 2]] = np_fill
    o3_src[1, 0:2, [1, 2]] = o3_fill
    np.testing.assert_equal(o3_src.cpu().numpy(), np_src)

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3d.core.Tensor(np_src, device=device)
    fill_shape = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]].shape
    np_fill_val = np.random.randint(5000, size=fill_shape).astype(np_src.dtype)
    o3_fill_val = o3d.core.Tensor(np_fill_val)
    o3_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = o3_fill_val
    np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = np_fill_val
    np.testing.assert_equal(o3_src.cpu().numpy(), np_src)


@pytest.mark.parametrize("np_func_name,o3_func_name", [("sqrt", "sqrt"),
                                                       ("sin", "sin"),
                                                       ("cos", "cos"),
                                                       ("negative", "neg"),
                                                       ("exp", "exp"),
                                                       ("abs", "abs"),
                                                       ("floor", "floor"),
                                                       ("ceil", "ceil"),
                                                       ("round", "round"),
                                                       ("trunc", "trunc")])
@pytest.mark.parametrize("device", list_devices())
def test_unary_elementwise(np_func_name, o3_func_name, device):
    np_t = np.array([-3.4, -2.6, -1.5, 0, 1.4, 2.6, 3.5]).astype(np.float32)
    o3_t = o3d.core.Tensor(np_t, device=device)

    # Test non-in-place version
    np.seterr(invalid='ignore')  # e.g. sqrt of negative should be -nan
    np.testing.assert_allclose(getattr(o3_t, o3_func_name)().cpu().numpy(),
                               getattr(np, np_func_name)(np_t),
                               rtol=1e-7,
                               atol=1e-7)

    # Test in-place version
    if o3_func_name not in ["floor", "ceil", "round", "trunc"]:
        o3_func_name_inplace = o3_func_name + "_"
        getattr(o3_t, o3_func_name_inplace)()
        np.testing.assert_allclose(o3_t.cpu().numpy(),
                                   getattr(np, np_func_name)(np_t),
                                   rtol=1e-7,
                                   atol=1e-7)


@pytest.mark.parametrize("device", list_devices())
def test_logical_ops(device):
    np_a = np.array([True, False, True, False])
    np_b = np.array([True, True, False, False])
    o3_a = o3d.core.Tensor(np_a, device=device)
    o3_b = o3d.core.Tensor(np_b, device=device)

    o3_r = o3_a.logical_and(o3_b)
    np_r = np.logical_and(np_a, np_b)
    np.testing.assert_equal(o3_r.cpu().numpy(), np_r)

    o3_r = o3_a.logical_or(o3_b)
    np_r = np.logical_or(np_a, np_b)
    np.testing.assert_equal(o3_r.cpu().numpy(), np_r)

    o3_r = o3_a.logical_xor(o3_b)
    np_r = np.logical_xor(np_a, np_b)
    np.testing.assert_equal(o3_r.cpu().numpy(), np_r)


@pytest.mark.parametrize("device", list_devices())
def test_comparision_ops(device):
    np_a = np.array([0, 1, -1])
    np_b = np.array([0, 0, 0])
    o3_a = o3d.core.Tensor(np_a, device=device)
    o3_b = o3d.core.Tensor(np_b, device=device)

    np.testing.assert_equal((o3_a > o3_b).cpu().numpy(), np_a > np_b)
    np.testing.assert_equal((o3_a >= o3_b).cpu().numpy(), np_a >= np_b)
    np.testing.assert_equal((o3_a < o3_b).cpu().numpy(), np_a < np_b)
    np.testing.assert_equal((o3_a <= o3_b).cpu().numpy(), np_a <= np_b)
    np.testing.assert_equal((o3_a == o3_b).cpu().numpy(), np_a == np_b)
    np.testing.assert_equal((o3_a != o3_b).cpu().numpy(), np_a != np_b)


@pytest.mark.parametrize("device", list_devices())
def test_non_zero(device):
    np_x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    np_nonzero_tuple = np.nonzero(np_x)
    o3_x = o3d.core.Tensor(np_x, device=device)
    o3_nonzero_tuple = o3_x.nonzero(as_tuple=True)
    for np_t, o3_t in zip(np_nonzero_tuple, o3_nonzero_tuple):
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_boolean_advanced_indexing(device):
    np_a = np.array([1, -1, -2, 3])
    o3_a = o3d.core.Tensor(np_a, device=device)
    np_a[np_a < 0] = 0
    o3_a[o3_a < 0] = 0
    np.testing.assert_equal(np_a, o3_a.cpu().numpy())

    np_x = np.array([[0, 1], [1, 1], [2, 2]])
    np_row_sum = np.array([1, 2, 4])
    np_y = np_x[np_row_sum <= 2, :]
    o3_x = o3d.core.Tensor(np_x, device=device)
    o3_row_sum = o3d.core.Tensor(np_row_sum)
    o3_y = o3_x[o3_row_sum <= 2, :]
    np.testing.assert_equal(np_y, o3_y.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_scalar_op(device):
    # +
    a = o3d.core.Tensor.ones((2, 3), o3d.core.Dtype.Float32, device=device)
    b = a.add(1)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = a + 1
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = 1 + a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = a + True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))

    # +=
    a = o3d.core.Tensor.ones((2, 3), o3d.core.Dtype.Float32, device=device)
    a.add_(1)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2))
    a += 1
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 3))
    a += True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 4))

    # -
    a = o3d.core.Tensor.ones((2, 3), o3d.core.Dtype.Float32, device=device)
    b = a.sub(1)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))
    b = a - 1
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))
    b = 10 - a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 9))
    b = a - True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))

    # -=
    a = o3d.core.Tensor.ones((2, 3), o3d.core.Dtype.Float32, device=device)
    a.sub_(1)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 0))
    a -= 1
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), -1))
    a -= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), -2))

    # *
    a = o3d.core.Tensor.full((2, 3), 2, o3d.core.Dtype.Float32, device=device)
    b = a.mul(10)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = a * 10
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = 10 * a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = a * True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))

    # *=
    a = o3d.core.Tensor.full((2, 3), 2, o3d.core.Dtype.Float32, device=device)
    a.mul_(10)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 20))
    a *= 10
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 200))
    a *= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 200))

    # /
    a = o3d.core.Tensor.full((2, 3), 20, o3d.core.Dtype.Float32, device=device)
    b = a.div(2)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 10))
    b = a / 2
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 10))
    b = a // 2
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 10))
    b = 10 / a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0.5))
    b = 10 // a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0.5))
    b = a / True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))

    # /=
    a = o3d.core.Tensor.full((2, 3), 20, o3d.core.Dtype.Float32, device=device)
    a.div_(2)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 10))
    a /= 2
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 5))
    a //= 2
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2.5))
    a /= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2.5))

    # logical_and
    a = o3d.core.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_and(True).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_and(5).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_and(False).cpu().numpy(), np.array([False, False]))
    np.testing.assert_equal(
        a.logical_and(0).cpu().numpy(), np.array([False, False]))

    # logical_and_
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_and_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_and_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_and_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False]))
    a.logical_and_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False]))

    # logical_or
    a = o3d.core.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_or(True).cpu().numpy(), np.array([True, True]))
    np.testing.assert_equal(
        a.logical_or(5).cpu().numpy(), np.array([True, True]))
    np.testing.assert_equal(
        a.logical_or(False).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_or(0).cpu().numpy(), np.array([True, False]))

    # logical_or_
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_or_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_or_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_or_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a.logical_or_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))

    # logical_xor
    a = o3d.core.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_xor(True).cpu().numpy(), np.array([False, True]))
    np.testing.assert_equal(
        a.logical_xor(5).cpu().numpy(), np.array([False, True]))
    np.testing.assert_equal(
        a.logical_xor(False).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_xor(0).cpu().numpy(), np.array([True, False]))

    # logical_xor_
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_xor_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_xor_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True]))
    a = o3d.core.Tensor([True, False], device=device)
    a.logical_xor_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a.logical_xor_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))

    # gt
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.gt(0)).cpu().numpy(),
                            np.array([False, False, True]))
    np.testing.assert_equal((a > 0).cpu().numpy(),
                            np.array([False, False, True]))

    # gt_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.gt_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False, True]))

    # lt
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.lt(0)).cpu().numpy(),
                            np.array([True, False, False]))
    np.testing.assert_equal((a < 0).cpu().numpy(),
                            np.array([True, False, False]))

    # lt_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.lt_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False, False]))

    # ge
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.ge(0)).cpu().numpy(),
                            np.array([False, True, True]))
    np.testing.assert_equal((a >= 0).cpu().numpy(),
                            np.array([False, True, True]))

    # ge_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.ge_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True, True]))

    # le
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.le(0)).cpu().numpy(),
                            np.array([True, True, False]))
    np.testing.assert_equal((a <= 0).cpu().numpy(),
                            np.array([True, True, False]))

    # le_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.le_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True, False]))

    # eq
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.eq(0)).cpu().numpy(),
                            np.array([False, True, False]))
    np.testing.assert_equal((a == 0).cpu().numpy(),
                            np.array([False, True, False]))

    # eq_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.eq_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True, False]))

    # ne
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.ne(0)).cpu().numpy(),
                            np.array([True, False, True]))
    np.testing.assert_equal((a != 0).cpu().numpy(),
                            np.array([True, False, True]))

    # ne_
    a = o3d.core.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.ne_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False, True]))


@pytest.mark.parametrize("device", list_devices())
def test_all_any(device):
    a = o3d.core.Tensor([False, True, True, True],
                        dtype=o3d.core.Dtype.Bool,
                        device=device)
    assert not a.all()
    assert a.any()

    a = o3d.core.Tensor([True, True, True, True],
                        dtype=o3d.core.Dtype.Bool,
                        device=device)
    assert a.all()

    # Empty
    a = o3d.core.Tensor([], dtype=o3d.core.Dtype.Bool, device=device)
    assert a.all()
    assert not a.any()


@pytest.mark.parametrize("device", list_devices())
def test_allclose_isclose(device):
    a = o3d.core.Tensor([1, 2], device=device)
    b = o3d.core.Tensor([1, 3], device=device)
    assert not a.allclose(b)
    np.testing.assert_allclose(
        a.isclose(b).cpu().numpy(), np.array([True, False]))

    assert a.allclose(b, atol=1)
    np.testing.assert_allclose(
        a.isclose(b, atol=1).cpu().numpy(), np.array([True, True]))

    # Test cases from
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    a = o3d.core.Tensor([1e10, 1e-7], device=device)
    b = o3d.core.Tensor([1.00001e10, 1e-8], device=device)
    assert not a.allclose(b)
    a = o3d.core.Tensor([1e10, 1e-8], device=device)
    b = o3d.core.Tensor([1.00001e10, 1e-9], device=device)
    assert a.allclose(b)
    a = o3d.core.Tensor([1e10, 1e-8], device=device)
    b = o3d.core.Tensor([1.0001e10, 1e-9], device=device)
    assert not a.allclose(b)


@pytest.mark.parametrize("device", list_devices())
def test_issame(device):
    dtype = o3d.core.Dtype.Float32
    a = o3d.core.Tensor.ones((2, 3), dtype, device=device)
    b = o3d.core.Tensor.ones((2, 3), dtype, device=device)
    assert a.allclose(b)
    assert not a.issame(b)

    c = a
    assert a.allclose(c)
    assert a.issame(c)

    d = a[:, 0:2]
    e = a[:, 0:2]
    assert d.allclose(e)
    assert d.issame(e)


@pytest.mark.parametrize("device", list_devices())
def test_item(device):
    o3_t = o3d.core.Tensor.ones(
        (2, 3), dtype=o3d.core.Dtype.Float32, device=device) * 1.5
    assert o3_t[0, 0].item() == 1.5
    assert isinstance(o3_t[0, 0].item(), float)

    o3_t = o3d.core.Tensor.ones(
        (2, 3), dtype=o3d.core.Dtype.Float64, device=device) * 1.5
    assert o3_t[0, 0].item() == 1.5
    assert isinstance(o3_t[0, 0].item(), float)

    o3_t = o3d.core.Tensor.ones(
        (2, 3), dtype=o3d.core.Dtype.Int32, device=device) * 1.5
    assert o3_t[0, 0].item() == 1
    assert isinstance(o3_t[0, 0].item(), int)

    o3_t = o3d.core.Tensor.ones(
        (2, 3), dtype=o3d.core.Dtype.Int64, device=device) * 1.5
    assert o3_t[0, 0].item() == 1
    assert isinstance(o3_t[0, 0].item(), int)

    o3_t = o3d.core.Tensor.ones((2, 3),
                                dtype=o3d.core.Dtype.Bool,
                                device=device)
    assert o3_t[0, 0].item() == True
    assert isinstance(o3_t[0, 0].item(), bool)


@pytest.mark.parametrize("device", list_devices())
def test_save_load(device):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/tensor.npy"

        o3_tensors = [
            o3d.core.Tensor([[1, 2], [3, 4]],
                            dtype=o3d.core.Dtype.Float32,
                            device=device),
            o3d.core.Tensor(3.14, dtype=o3d.core.Dtype.Float32, device=device),
            o3d.core.Tensor.ones((0,),
                                 dtype=o3d.core.Dtype.Float32,
                                 device=device),
            o3d.core.Tensor.ones((0, 0),
                                 dtype=o3d.core.Dtype.Float32,
                                 device=device),
            o3d.core.Tensor.ones((0, 1, 0),
                                 dtype=o3d.core.Dtype.Float32,
                                 device=device)
        ]
        np_tensors = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array(3.14, dtype=np.float32),
            np.ones((0,), dtype=np.float32),
            np.ones((0, 0), dtype=np.float32),
            np.ones((0, 1, 0), dtype=np.float32)
        ]
        for o3_t, np_t in zip(o3_tensors, np_tensors):
            # Open3D -> Numpy.
            o3_t.save(file_name)
            o3_t_load = o3d.core.Tensor.load(file_name)
            np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)

            # Open3D -> Numpy.
            np_t_load = np.load(file_name)
            np.testing.assert_equal(np_t_load, np_t_load)

            # Numpy -> Open3D.
            np.save(file_name, np_t)
            o3_t_load = o3d.core.Tensor.load(file_name)
            np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)

        # Ragged tensor: exception.
        np_t = np.array([[1, 2, 3], [4, 5]], dtype=np.dtype(object))
        np.save(file_name, np_t)
        with pytest.raises(RuntimeError):
            o3_t_load = o3d.core.Tensor.load(file_name)

        # Fortran order: exception.
        np_t = np.array([[1, 2, 3], [4, 5, 6]])
        np_t = np.asfortranarray(np_t)
        np.save(file_name, np_t)
        with pytest.raises(RuntimeError):
            o3_t_load = o3d.core.Tensor.load(file_name)

        # Unsupported dtype: exception.
        np_t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        np.save(file_name, np_t)
        with pytest.raises(RuntimeError):
            o3_t_load = o3d.core.Tensor.load(file_name)

        # Non-contiguous numpy array.
        np_t = np.arange(24).reshape(2, 3, 4)
        assert np_t.flags['C_CONTIGUOUS']
        np_t = np_t[0:2:1, 0:3:2, 0:4:2]
        assert not np_t.flags['C_CONTIGUOUS']
        np.save(file_name, np_t)
        o3_t_load = o3d.core.Tensor.load(file_name)
        assert o3_t_load.is_contiguous()
        np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)
