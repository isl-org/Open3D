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
import pytest

try:
    import torch
    import torch.utils.dlpack
except ImportError:
    _torch_imported = False
else:
    _torch_imported = True


def list_devices():
    devices = [o3d.Device("CPU:" + str(0))]
    if _torch_imported:
        if (o3d.cuda.device_count() != torch.cuda.device_count()):
            raise RuntimeError(
                f"o3d.cuda.device_count() != torch.cuda.device_count(), "
                f"{o3d.cuda.device_count()} != {torch.cuda.device_count()}")
    else:
        print(
            'Warning: torch is not imported, cannot guarantee the correctness of device_count()'
        )

    for i in range(o3d.cuda.device_count()):
        devices.append(o3d.Device("CUDA:" + str(i)))
    return devices


def test_dtype():
    dtype = o3d.Dtype.Int32
    assert o3d.DtypeUtil.byte_size(dtype) == 4
    assert f"{dtype}" == "Dtype.Int32"


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

    assert o3d.Device("CUDA", 1).__str__() == "CUDA:1"


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
    no3_t = o3d.Tensor(li_t, dtype, device)
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

    # Boolean
    np_t = np.array([True, False, True], dtype=np.bool)
    o3_t = o3d.Tensor([True, False, True], o3d.Dtype.Bool, device)
    np.testing.assert_equal(np_t, o3_t.numpy())
    o3_t = o3d.Tensor(np_t, o3d.Dtype.Bool, device)
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
    if not _torch_imported:
        return

    src_t = np.array([[10, 11, 12.], [13., 14., 15.]])

    def get_dst_t():
        o3d_t = o3d.Tensor(src_t, device=device)  # Copy
        dst_t = torch.utils.dlpack.from_dlpack(o3d_t.to_dlpack())
        return dst_t

    dst_t = get_dst_t().cpu().numpy()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("device", list_devices())
def test_tensor_from_to_pytorch(device):
    if not _torch_imported:
        return

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
    if not _torch_imported:
        return

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


def test_binary_ew_ops():
    a = o3d.Tensor(np.array([4, 6, 8, 10, 12, 14]))
    b = o3d.Tensor(np.array([2, 3, 4, 5, 6, 7]))
    np.testing.assert_equal((a + b).numpy(), np.array([6, 9, 12, 15, 18, 21]))
    np.testing.assert_equal((a - b).numpy(), np.array([2, 3, 4, 5, 6, 7]))
    np.testing.assert_equal((a * b).numpy(), np.array([8, 18, 32, 50, 72, 98]))
    np.testing.assert_equal((a / b).numpy(), np.array([2, 2, 2, 2, 2, 2]))

    a = o3d.Tensor(np.array([4, 6, 8, 10, 12, 14]))
    a += b
    np.testing.assert_equal(a.numpy(), np.array([6, 9, 12, 15, 18, 21]))

    a = o3d.Tensor(np.array([4, 6, 8, 10, 12, 14]))
    a -= b
    np.testing.assert_equal(a.numpy(), np.array([2, 3, 4, 5, 6, 7]))

    a = o3d.Tensor(np.array([4, 6, 8, 10, 12, 14]))
    a *= b
    np.testing.assert_equal(a.numpy(), np.array([8, 18, 32, 50, 72, 98]))

    a = o3d.Tensor(np.array([4, 6, 8, 10, 12, 14]))
    a //= b
    np.testing.assert_equal(a.numpy(), np.array([2, 2, 2, 2, 2, 2]))


def test_to():
    a = o3d.Tensor(np.array([0.1, 1.2, 2.3, 3.4, 4.5, 5.6]).astype(np.float32))
    b = a.to(o3d.Dtype.Int32)
    np.testing.assert_equal(b.numpy(), np.array([0, 1, 2, 3, 4, 5]))
    assert b.shape == o3d.SizeVector([6])
    assert b.strides == o3d.SizeVector([1])
    assert b.dtype == o3d.Dtype.Int32
    assert b.device == a.device


def test_unary_ew_ops():
    src_vals = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    src = o3d.Tensor(src_vals)

    rtol = 1e-5
    atol = 0
    np.testing.assert_allclose(src.sqrt().numpy(),
                               np.sqrt(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.sin().numpy(),
                               np.sin(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.cos().numpy(),
                               np.cos(src_vals),
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.neg().numpy(),
                               -src_vals,
                               rtol=rtol,
                               atol=atol)
    np.testing.assert_allclose(src.exp().numpy(),
                               np.exp(src_vals),
                               rtol=rtol,
                               atol=atol)


def test_tensorlist_operations():
    a = o3d.TensorList([3, 4], o3d.Dtype.Float32, o3d.Device(), size=1)
    assert a.size() == 1

    b = o3d.TensorList.from_tensor(
        o3d.Tensor(np.ones((2, 3, 4), dtype=np.float32)))
    assert b.size() == 2

    c = o3d.TensorList.from_tensors(
        [o3d.Tensor(np.zeros((3, 4), dtype=np.float32))])
    assert c.size() == 1

    d = a + b
    assert d.size() == 3

    e = o3d.TensorList.concat(c, d)
    assert e.size() == 4

    e.push_back(o3d.Tensor(np.zeros((3, 4), dtype=np.float32)))
    assert e.size() == 5

    e.extend(d)
    assert e.size() == 8

    e += a
    assert e.size() == 9


def test_getitem():
    np_t = np.array(range(24)).reshape((2, 3, 4))
    o3_t = o3d.Tensor(np_t)

    np.testing.assert_equal(o3_t[:].numpy(), np_t[:])
    np.testing.assert_equal(o3_t[0].numpy(), np_t[0])
    np.testing.assert_equal(o3_t[0, 1].numpy(), np_t[0, 1])
    np.testing.assert_equal(o3_t[0, :].numpy(), np_t[0, :])
    np.testing.assert_equal(o3_t[0, 1:3].numpy(), np_t[0, 1:3])
    np.testing.assert_equal(o3_t[0, :, :-2].numpy(), np_t[0, :, :-2])
    np.testing.assert_equal(o3_t[0, 1:3, 2].numpy(), np_t[0, 1:3, 2])
    np.testing.assert_equal(o3_t[0, 1:-1, 2].numpy(), np_t[0, 1:-1, 2])
    np.testing.assert_equal(o3_t[0, 1:3, 0:4:2].numpy(), np_t[0, 1:3, 0:4:2])
    np.testing.assert_equal(o3_t[0, 1:3, 0:-1:2].numpy(), np_t[0, 1:3, 0:-1:2])
    np.testing.assert_equal(o3_t[0, 1, :].numpy(), np_t[0, 1, :])

    # Slice the slice
    np.testing.assert_equal(o3_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3].numpy(),
                            np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3])


def test_setitem():
    np_ref = np.array(range(24)).reshape((2, 3, 4))
    o3_ref = o3d.Tensor(np_ref)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[:].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[:] = np_fill_t
    o3_t[:] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0] = np_fill_t
    o3_t[0] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1] = np_fill_t
    o3_t[0, 1] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, :].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, :] = np_fill_t
    o3_t[0, :] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1:3].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1:3] = np_fill_t
    o3_t[0, 1:3] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, :, :-2].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, :, :-2] = np_fill_t
    o3_t[0, :, :-2] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 2].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1:3, 2] = np_fill_t
    o3_t[0, 1:3, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1:-1, 2].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1:-1, 2] = np_fill_t
    o3_t[0, 1:-1, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:4:2].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1:3, 0:4:2] = np_fill_t
    o3_t[0, 1:3, 0:4:2] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:-1:2].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1:3, 0:-1:2] = np_fill_t
    o3_t[0, 1:3, 0:-1:2] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0, 1, :].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0, 1, :] = np_fill_t
    o3_t[0, 1, :] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3d.Tensor(np_t)
    np_fill_t = np.random.rand(*np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3].shape)
    o3_fill_t = o3d.Tensor(np_fill_t)
    np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = np_fill_t
    o3_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = o3_fill_t
    np.testing.assert_equal(o3_t.numpy(), np_t)


def test_cast_to_py_tensor():
    a = o3d.Tensor([1])
    b = o3d.Tensor([2])
    c = a + b
    assert isinstance(c, o3d.Tensor)  # Not o3d.open3d-pybind.Tensor


def test_advanced_index_get_mixed():
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.Tensor(np_src)

    np_dst = np_src[1, 0:2, [1, 2]]
    o3_dst = o3_src[1, 0:2, [1, 2]]
    np.testing.assert_equal(o3_dst.numpy(), np_dst)

    # Subtle differences between slice and list
    np_src = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800]).reshape(3, 3)
    o3_src = o3d.Tensor(np_src)
    np.testing.assert_equal(o3_src[1, 2].numpy(), np_src[1, 2])
    np.testing.assert_equal(o3_src[[1, 2]].numpy(), np_src[[1, 2]])
    np.testing.assert_equal(o3_src[(1, 2)].numpy(), np_src[(1, 2)])
    np.testing.assert_equal(o3_src[(1, 2), [1, 2]].numpy(),
                            np_src[(1, 2), [1, 2]])

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3d.Tensor(np_src)
    o3_dst = o3_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np_dst = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np.testing.assert_equal(o3_dst.numpy(), np_dst)


def test_advanced_index_set_mixed():
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3d.Tensor(np_src)

    np_fill = np.array(([[100, 200], [300, 400]]))
    o3_fill = o3d.Tensor(np_fill)

    np_src[1, 0:2, [1, 2]] = np_fill
    o3_src[1, 0:2, [1, 2]] = o3_fill
    np.testing.assert_equal(o3_src.numpy(), np_src)

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3d.Tensor(np_src)
    fill_shape = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]].shape
    np_fill_val = np.random.randint(5000, size=fill_shape).astype(np_src.dtype)
    o3_fill_val = o3d.Tensor(np_fill_val)
    o3_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = o3_fill_val
    np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = np_fill_val
    np.testing.assert_equal(o3_src.numpy(), np_src)


def test_tensorlist_indexing():
    # 5 x (3, 4)
    dtype = o3d.Dtype.Float32
    device = o3d.Device("CPU:0")
    np_t = np.ones((5, 3, 4), dtype=np.float32)
    t = o3d.Tensor(np_t, dtype, device)

    tl = o3d.TensorList.from_tensor(t, inplace=True)

    # set slices [1, 3]
    tl.tensor()[1:5:2] = o3d.Tensor(3 * np.ones((2, 3, 4), dtype=np.float32))

    # set items [4]
    tl[-1] = o3d.Tensor(np.zeros((3, 4), dtype=np.float32))

    # get items
    np.testing.assert_allclose(tl[0].numpy(), np.ones((3, 4), dtype=np.float32))
    np.testing.assert_allclose(tl[1].numpy(), 3 * np.ones(
        (3, 4), dtype=np.float32))
    np.testing.assert_allclose(tl[2].numpy(), np.ones((3, 4), dtype=np.float32))
    np.testing.assert_allclose(tl[3].numpy(), 3 * np.ones(
        (3, 4), dtype=np.float32))
    np.testing.assert_allclose(tl[4].numpy(), np.zeros((3, 4),
                                                       dtype=np.float32))

    # push_back
    tl.push_back(o3d.Tensor(-1 * np.ones((3, 4)), dtype, device))
    assert tl.size() == 6
    np.testing.assert_allclose(tl[5].numpy(), -1 * np.ones(
        (3, 4), dtype=np.float32))

    tl += tl
    assert tl.size() == 12
    for offset in [0, 6]:
        np.testing.assert_allclose(tl[0 + offset].numpy(),
                                   np.ones((3, 4), dtype=np.float32))
        np.testing.assert_allclose(tl[1 + offset].numpy(), 3 * np.ones(
            (3, 4), dtype=np.float32))
        np.testing.assert_allclose(tl[2 + offset].numpy(),
                                   np.ones((3, 4), dtype=np.float32))
        np.testing.assert_allclose(tl[3 + offset].numpy(), 3 * np.ones(
            (3, 4), dtype=np.float32))
        np.testing.assert_allclose(tl[4 + offset].numpy(),
                                   np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize("device", list_devices())
def test_tensor_from_to_pytorch(device):
    if not _torch_imported:
        return


@pytest.mark.parametrize("np_func_name,o3_func_name", [("sqrt", "sqrt"),
                                                       ("sin", "sin"),
                                                       ("cos", "cos"),
                                                       ("negative", "neg"),
                                                       ("exp", "exp"),
                                                       ("abs", "abs")])
def test_unary_elementwise(np_func_name, o3_func_name):
    np_t = np.array([-3, -2, -1, 9, 1, 2, 3]).astype(np.float32)
    o3_t = o3d.Tensor(np_t)

    # Test non-in-place version
    np.seterr(invalid='ignore')  # e.g. sqrt of negative should be -nan
    np.testing.assert_allclose(
        getattr(o3_t, o3_func_name)().numpy(),
        getattr(np, np_func_name)(np_t))

    # Test in-place version
    o3_func_name_inplace = o3_func_name + "_"
    getattr(o3_t, o3_func_name_inplace)()
    np.testing.assert_allclose(o3_t.numpy(), getattr(np, np_func_name)(np_t))


def test_logical_ops():
    np_a = np.array([True, False, True, False])
    np_b = np.array([True, True, False, False])
    o3_a = o3d.Tensor(np_a)
    o3_b = o3d.Tensor(np_b)

    o3_r = o3_a.logical_and(o3_b)
    np_r = np.logical_and(np_a, np_b)
    np.testing.assert_equal(o3_r.numpy(), np_r)

    o3_r = o3_a.logical_or(o3_b)
    np_r = np.logical_or(np_a, np_b)
    np.testing.assert_equal(o3_r.numpy(), np_r)

    o3_r = o3_a.logical_xor(o3_b)
    np_r = np.logical_xor(np_a, np_b)
    np.testing.assert_equal(o3_r.numpy(), np_r)


def test_comparision_ops():
    np_a = np.array([0, 1, -1])
    np_b = np.array([0, 0, 0])
    o3_a = o3d.Tensor(np_a)
    o3_b = o3d.Tensor(np_b)

    np.testing.assert_equal((o3_a > o3_b).numpy(), np_a > np_b)
    np.testing.assert_equal((o3_a >= o3_b).numpy(), np_a >= np_b)
    np.testing.assert_equal((o3_a < o3_b).numpy(), np_a < np_b)
    np.testing.assert_equal((o3_a <= o3_b).numpy(), np_a <= np_b)
    np.testing.assert_equal((o3_a == o3_b).numpy(), np_a == np_b)
    np.testing.assert_equal((o3_a != o3_b).numpy(), np_a != np_b)
