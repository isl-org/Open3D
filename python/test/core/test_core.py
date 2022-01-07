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
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices


def list_dtypes():
    return [
        o3c.float32,
        o3c.float64,
        o3c.int8,
        o3c.int16,
        o3c.int32,
        o3c.int64,
        o3c.uint8,
        o3c.uint16,
        o3c.uint32,
        o3c.uint64,
        o3c.bool,
    ]


def list_non_bool_dtypes():
    return [
        o3c.float32,
        o3c.float64,
        o3c.int8,
        o3c.int16,
        o3c.int32,
        o3c.int64,
        o3c.uint8,
        o3c.uint16,
        o3c.uint32,
        o3c.uint64,
    ]


def to_numpy_dtype(dtype: o3c.Dtype):
    conversions = {
        o3c.float32: np.float32,
        o3c.float64: np.float64,
        o3c.int8: np.int8,
        o3c.int16: np.int16,
        o3c.int32: np.int32,
        o3c.int64: np.int64,
        o3c.uint8: np.uint8,
        o3c.uint16: np.uint16,
        o3c.uint32: np.uint32,
        o3c.uint64: np.uint64,
        o3c.bool8: np.bool8,  # np.bool deprecated
        o3c.bool: np.bool8,  # o3c.bool is an alias for o3c.bool8
    }
    return conversions[dtype]


@pytest.mark.parametrize("dtype", list_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_creation(dtype, device):
    # Shape takes tuple, list or o3c.SizeVector
    t = o3c.Tensor.empty((2, 3), dtype, device=device)
    assert t.shape == o3c.SizeVector([2, 3])
    t = o3c.Tensor.empty([2, 3], dtype, device=device)
    assert t.shape == o3c.SizeVector([2, 3])
    t = o3c.Tensor.empty(o3c.SizeVector([2, 3]), dtype, device=device)
    assert t.shape == o3c.SizeVector([2, 3])

    # Test zeros and ones
    t = o3c.Tensor.zeros((2, 3), dtype, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.zeros((2, 3), dtype=np.float32))
    t = o3c.Tensor.ones((2, 3), dtype, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.ones((2, 3), dtype=np.float32))

    # Automatic casting of dtype.
    t = o3c.Tensor.full((2,), False, o3c.float32, device=device)
    np.testing.assert_equal(t.cpu().numpy(),
                            np.full((2,), False, dtype=np.float32))
    t = o3c.Tensor.full((2,), 3.5, o3c.uint8, device=device)
    np.testing.assert_equal(t.cpu().numpy(), np.full((2,), 3.5, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(), (0,), (1,), (0, 2), (0, 0, 2),
                                   (2, 0, 3)])
@pytest.mark.parametrize("dtype", list_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_creation_special_shapes(shape, dtype, device):
    o3_t = o3c.Tensor.full(shape, 3.14, dtype, device=device)
    np_t = np.full(shape, 3.14, dtype=to_numpy_dtype(dtype))
    np.testing.assert_allclose(o3_t.cpu().numpy(), np_t)


def test_dtype():
    dtype = o3c.int32
    assert dtype.byte_size() == 4
    assert "{}".format(dtype) == "Int32"


def test_device():
    device = o3c.Device()
    assert device.get_type() == o3c.Device.DeviceType.CPU
    assert device.get_id() == 0

    device = o3c.Device("CUDA", 1)
    assert device.get_type() == o3c.Device.DeviceType.CUDA
    assert device.get_id() == 1

    device = o3c.Device("CUDA:2")
    assert device.get_type() == o3c.Device.DeviceType.CUDA
    assert device.get_id() == 2

    assert o3c.Device("CUDA", 1) == o3c.Device("CUDA:1")
    assert o3c.Device("CUDA", 1) != o3c.Device("CUDA:0")

    assert o3c.Device("CUDA", 1).__str__() == "CUDA:1"


@pytest.mark.parametrize("dtype", list_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_tensor_constructor(dtype, device):
    # Numpy array
    np_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=to_numpy_dtype(dtype))
    o3_t = o3c.Tensor(np_t, device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # 2D list
    li_t = [[0, 1, 2], [3, 4, 5]]
    np_t = np.array(li_t, dtype=to_numpy_dtype(dtype))
    o3_t = o3c.Tensor(li_t, dtype, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # 2D list, inconsistent length
    li_t = [[0, 1, 2], [3, 4]]
    with pytest.raises(Exception):
        # Suppress inconsistent length warning as this check is intentional
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=np.VisibleDeprecationWarning)
            o3_t = o3c.Tensor(li_t, dtype, device)

    # Automatic casting
    np_t_double = np.array([[0., 1.5, 2.], [3., 4., 5.]])
    np_t_int = np.array([[0, 1, 2], [3, 4, 5]])
    o3_t = o3c.Tensor(np_t_double, o3c.int32, device)
    np.testing.assert_equal(np_t_int, o3_t.cpu().numpy())

    # Special strides
    np_t = np.random.randint(10, size=(10, 10))[1:10:2, 1:10:3].T
    o3_t = o3c.Tensor(np_t, o3c.int32, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Boolean
    np_t = np.array([True, False, True], dtype=np.bool8)
    o3_t = o3c.Tensor([True, False, True], o3c.bool, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())
    o3_t = o3c.Tensor(np_t, o3c.bool, device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Scalar Boolean
    np_t = np.array(True)
    o3_t = o3c.Tensor(True, dtype=None, device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())
    o3_t = o3c.Tensor(True, dtype=o3c.bool, device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_arange(device):
    # Full parameters.
    setups = [(0, 10, 1), (0, 10, 1), (0.0, 10.0, 2.0), (0.0, -10.0, -2.0)]
    for start, stop, step in setups:
        np_t = np.arange(start, stop, step)
        o3_t = o3c.Tensor.arange(start, stop, step, dtype=None, device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Only stop.
    for stop in [1.0, 2.0, 3.0, 1, 2, 3]:
        np_t = np.arange(stop)
        o3_t = o3c.Tensor.arange(stop, dtype=None, device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Only start, stop (step = 1).
    setups = [(0, 10), (0, 10), (0.0, 10.0), (0.0, -10.0)]
    for start, stop in setups:
        np_t = np.arange(start, stop)
        # Not full parameter list, need to specify device by kw.
        o3_t = o3c.Tensor.arange(start, stop, dtype=None, device=device)
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: int -> int.
    o3_t = o3c.Tensor.arange(0, 5, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3c.int64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: int, float -> float.
    o3_t = o3c.Tensor.arange(0, 5.0, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3c.float64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: float, float -> float.
    o3_t = o3c.Tensor.arange(0.0, 5.0, dtype=None, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3c.float64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    # Type inference: explicit type.
    o3_t = o3c.Tensor.arange(0.0, 5.0, dtype=o3c.int64, device=device)
    np_t = np.arange(0, 5)
    assert o3_t.dtype == o3c.int64
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_flatten(device):

    # Flatten 0-D tensor
    src_t = o3c.Tensor(3, dtype=o3c.Dtype.Int64).to(device)
    dst_t = o3c.Tensor([3], dtype=o3c.Dtype.Int64).to(device)
    assert dst_t.allclose(src_t.flatten())
    assert dst_t.allclose(src_t.flatten(0))
    assert dst_t.allclose(src_t.flatten(-1))

    assert dst_t.allclose(src_t.flatten(0, 0))
    assert dst_t.allclose(src_t.flatten(0, -1))
    assert dst_t.allclose(src_t.flatten(-1, 0))
    assert dst_t.allclose(src_t.flatten(-1, -1))

    with pytest.raises(RuntimeError):
        src_t.flatten(-2)
    with pytest.raises(RuntimeError):
        src_t.flatten(1)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, -2)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, 1)

    # Flatten 1-D tensor
    src_t = o3c.Tensor([1, 2, 3], dtype=o3c.Dtype.Int64).to(device)
    dst_t = o3c.Tensor([1, 2, 3], dtype=o3c.Dtype.Int64).to(device)

    assert dst_t.allclose(src_t.flatten())
    assert dst_t.allclose(src_t.flatten(0))
    assert dst_t.allclose(src_t.flatten(-1))

    assert dst_t.allclose(src_t.flatten(0, 0))
    assert dst_t.allclose(src_t.flatten(0, -1))
    assert dst_t.allclose(src_t.flatten(-1, 0))
    assert dst_t.allclose(src_t.flatten(-1, -1))

    with pytest.raises(RuntimeError):
        src_t.flatten(-2)
    with pytest.raises(RuntimeError):
        src_t.flatten(1)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, -2)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, 1)

    # Flatten 2-D tensor
    src_t = o3c.Tensor([[1, 2, 3], [4, 5, 6]], dtype=o3c.Dtype.Int64).to(device)
    dst_t_flat = o3c.Tensor([1, 2, 3, 4, 5, 6],
                            dtype=o3c.Dtype.Int64).to(device)
    dst_t_unchanged = o3c.Tensor([[1, 2, 3], [4, 5, 6]],
                                 dtype=o3c.Dtype.Int64).to(device)

    assert dst_t_flat.allclose(src_t.flatten())
    assert dst_t_flat.allclose(src_t.flatten(0))
    assert dst_t_flat.allclose(src_t.flatten(-2))

    assert dst_t_flat.allclose(src_t.flatten(0, 1))
    assert dst_t_flat.allclose(src_t.flatten(-2, 1))
    assert dst_t_flat.allclose(src_t.flatten(0, -1))
    assert dst_t_flat.allclose(src_t.flatten(-2, -1))

    assert dst_t_unchanged.allclose(src_t.flatten(1))
    assert dst_t_unchanged.allclose(src_t.flatten(-1))

    for dim in range(-2, 2):
        assert dst_t_unchanged.allclose(src_t.flatten(dim, dim))

    # Out of bounds dimensions
    with pytest.raises(RuntimeError):
        src_t.flatten(0, 2)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, -3)
    with pytest.raises(RuntimeError):
        src_t.flatten(-3, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(2, 0)

    # end_dim is greater than start_dim
    with pytest.raises(RuntimeError):
        src_t.flatten(1, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(-1, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(1, -2)
    with pytest.raises(RuntimeError):
        src_t.flatten(-1, -2)

    # Flatten 3-D tensor
    src_t = o3c.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                       dtype=o3c.Dtype.Int64).to(device)
    dst_t_flat = o3c.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            dtype=o3c.Dtype.Int64).to(device)
    dst_t_unchanged = o3c.Tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=o3c.Dtype.Int64).to(device)
    dst_t_first_two_flat = o3c.Tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=o3c.Dtype.Int64).to(device)
    dst_t_last_two_flat = o3c.Tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        dtype=o3c.Dtype.Int64).to(device)

    assert dst_t_flat.allclose(src_t.flatten())
    assert dst_t_flat.allclose(src_t.flatten(0))
    assert dst_t_flat.allclose(src_t.flatten(-3))

    assert dst_t_flat.allclose(src_t.flatten(0, 2))
    assert dst_t_flat.allclose(src_t.flatten(-3, 2))
    assert dst_t_flat.allclose(src_t.flatten(0, -1))
    assert dst_t_flat.allclose(src_t.flatten(-3, -1))

    assert dst_t_first_two_flat.allclose(src_t.flatten(0, 1))
    assert dst_t_first_two_flat.allclose(src_t.flatten(0, -2))
    assert dst_t_first_two_flat.allclose(src_t.flatten(-3, 1))
    assert dst_t_first_two_flat.allclose(src_t.flatten(-3, -2))

    assert dst_t_last_two_flat.allclose(src_t.flatten(1, 2))
    assert dst_t_last_two_flat.allclose(src_t.flatten(1, -1))
    assert dst_t_last_two_flat.allclose(src_t.flatten(-2, 2))
    assert dst_t_last_two_flat.allclose(src_t.flatten(-2, -1))

    for dim in range(-3, 3):
        assert dst_t_unchanged.allclose(src_t.flatten(dim, dim))

    # Out of bounds dimensions
    with pytest.raises(RuntimeError):
        src_t.flatten(0, 3)
    with pytest.raises(RuntimeError):
        src_t.flatten(0, -4)
    with pytest.raises(RuntimeError):
        src_t.flatten(-4, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(3, 0)

    # end_dim is greater than start_dim
    with pytest.raises(RuntimeError):
        src_t.flatten(1, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(2, 0)
    with pytest.raises(RuntimeError):
        src_t.flatten(2, 0)


@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_append(dtype, device):
    # Appending 0-D.
    # 0-D can only be appended along axis = null.
    self = o3c.Tensor(0, dtype=dtype, device=device)
    values = o3c.Tensor(1, dtype=dtype, device=device)
    output_t = self.append(values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    with pytest.raises(
            RuntimeError,
            match=r"Zero-dimensional tensor can only be concatenated along "
            "axis = null, but got 0."):
        self.append(values, axis=0)

    # Appending 1-D.
    # 1-D can be appended along axis = 0, -1.
    self = o3c.Tensor([0, 1], dtype=dtype, device=device)
    values = o3c.Tensor([2, 3, 4], dtype=dtype, device=device)
    output_t = self.append(values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # axis must always be in range [-num_dims, num_dims).
    # So, 1-D tensor cannot be appended along axis 1 or -2.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 1, but it must satisfy -1 <= dim <= 0"
    ):
        self.append(values, axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -2, but it must satisfy -1 <= dim <= 0"
    ):
        self.append(values, axis=-2)

    # Appending 2-D. [2, 2] to [2, 2].
    # [2, 2] to [2, 2] can be appended along axis = 0, 1, -1, -2.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5], [6, 7]], dtype=dtype, device=device)

    output_t = self.append(values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=-1)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-1)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # axis must always be in range [-num_dims, num_dims).
    # So, 2-D tensor cannot be appended along axis 2 or -3.
    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == 2, but it must satisfy -2 <= dim <= 1"
    ):
        self.append(values, axis=2)

    with pytest.raises(
            RuntimeError,
            match=
            r"Index out-of-range: dim == -3, but it must satisfy -2 <= dim <= 1"
    ):
        self.append(values, axis=-3)

    # Appending 2-D. [1, 2] to [2, 2].
    # [1, 2] to [2, 2] can be appended along axis = 0, -2.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=dtype, device=device)
    values = o3c.Tensor([[4, 5]], dtype=dtype, device=device)

    output_t = self.append(values)
    output_np = np.append(arr=self.cpu().numpy(), values=values.cpu().numpy())

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=0)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=0)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    output_t = self.append(values, axis=-2)
    output_np = np.append(arr=self.cpu().numpy(),
                          values=values.cpu().numpy(),
                          axis=-2)

    np.testing.assert_equal(output_np, output_t.cpu().numpy())

    # all the dimensions other than the dimension along the axis, must be
    # exactly same.
    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        self.append(values, axis=1)

    with pytest.raises(
            RuntimeError,
            match=
            r"All the input tensor dimensions, other than dimension size along "
            "concatenation axis must be same, but along dimension 0, the "
            "tensor at index 0 has size 2 and the tensor at index 1 has size 1."
    ):
        self.append(values, axis=-1)

    # dtype and device must be same for all the input tensors.
    self = o3c.Tensor([[0, 1], [2, 3]], dtype=o3c.Dtype.Float32, device=device)
    values = o3c.Tensor([[4, 5]], dtype=o3c.Dtype.Float64, device=device)
    with pytest.raises(
            RuntimeError,
            match=r"Tensor has dtype Float64, but is expected to have Float32"):
        self.append(values)


def test_tensor_from_to_numpy():
    # a->b copy; b, c share memory
    a = np.ones((2, 2))
    b = o3c.Tensor(a)
    c = b.numpy()

    c[0, 1] = 200
    r = np.array([[1., 200.], [1., 1.]])
    np.testing.assert_equal(r, b.numpy())
    np.testing.assert_equal(r, c)

    # a, b, c share memory
    a = np.array([[1., 1.], [1., 1.]])
    b = o3c.Tensor.from_numpy(a)
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
    o3d_t = o3c.Tensor.from_numpy(src_t)  # Shared memory
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
        o3d_t = o3c.Tensor(src_t)  # Copy
        dst_t = o3d_t.numpy()
        return dst_t

    dst_t = get_dst_t()
    np.testing.assert_equal(dst_t, src_t)


@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_binary_ew_ops(dtype, device):
    a = o3c.Tensor(np.array([4, 6, 8, 10, 12, 14]), dtype=dtype, device=device)
    b = o3c.Tensor(np.array([2, 3, 4, 5, 6, 7]), dtype=dtype, device=device)
    np.testing.assert_equal((a + b).cpu().numpy(),
                            np.array([6, 9, 12, 15, 18, 21]))
    np.testing.assert_equal((a - b).cpu().numpy(), np.array([2, 3, 4, 5, 6, 7]))
    np.testing.assert_equal((a * b).cpu().numpy(),
                            np.array([8, 18, 32, 50, 72, 98]))
    np.testing.assert_equal((a / b).cpu().numpy(), np.array([2, 2, 2, 2, 2, 2]))

    a = o3c.Tensor(np.array([4, 6, 8, 10, 12, 14]), dtype=dtype, device=device)
    a += b
    np.testing.assert_equal(a.cpu().numpy(), np.array([6, 9, 12, 15, 18, 21]))

    a = o3c.Tensor(np.array([4, 6, 8, 10, 12, 14]), dtype=dtype, device=device)
    a -= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([2, 3, 4, 5, 6, 7]))

    a = o3c.Tensor(np.array([4, 6, 8, 10, 12, 14]), dtype=dtype, device=device)
    a *= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([8, 18, 32, 50, 72, 98]))

    a = o3c.Tensor(np.array([4, 6, 8, 10, 12, 14]), dtype=dtype, device=device)
    a //= b
    np.testing.assert_equal(a.cpu().numpy(), np.array([2, 2, 2, 2, 2, 2]))


@pytest.mark.parametrize("device", list_devices())
def test_to(device):
    a = o3c.Tensor(np.array([0.1, 1.2, 2.3, 3.4, 4.5, 5.6]).astype(np.float32),
                   device=device)
    b = a.to(o3c.int32)
    np.testing.assert_equal(b.cpu().numpy(), np.array([0, 1, 2, 3, 4, 5]))
    assert b.shape == o3c.SizeVector([6])
    assert b.strides == o3c.SizeVector([1])
    assert b.dtype == o3c.int32
    assert b.device == a.device


@pytest.mark.parametrize("device", list_devices())
def test_unary_ew_ops(device):
    src_vals = np.array([0, 1, 2, 3, 4, 5]).astype(np.float32)
    src = o3c.Tensor(src_vals, device=device)

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
    o3_t = o3c.Tensor(np_t, device=device)

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

    # Slice a zero-dim tensor
    with pytest.raises(RuntimeError,
                       match=r"Cannot slice a scalar \(0-dim\) tensor."):
        o3c.Tensor.ones((), device=device)[:]
    with pytest.raises(RuntimeError,
                       match=r"Cannot slice a scalar \(0-dim\) tensor."):
        o3c.Tensor.ones((), device=device)[0:1]


@pytest.mark.parametrize("device", list_devices())
def test_setitem(device):
    np_ref = np.array(range(24)).reshape((2, 3, 4))

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[:].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[:] = np_fill_t
    o3_t[:] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0] = np_fill_t
    o3_t[0] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1] = np_fill_t
    o3_t[0, 1] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, :].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, :] = np_fill_t
    o3_t[0, :] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1:3] = np_fill_t
    o3_t[0, 1:3] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, :, :-2].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, :, :-2] = np_fill_t
    o3_t[0, :, :-2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 2].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 2] = np_fill_t
    o3_t[0, 1:3, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:-1, 2].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1:-1, 2] = np_fill_t
    o3_t[0, 1:-1, 2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:4:2].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 0:4:2] = np_fill_t
    o3_t[0, 1:3, 0:4:2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1:3, 0:-1:2].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1:3, 0:-1:2] = np_fill_t
    o3_t[0, 1:3, 0:-1:2] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0, 1, :].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0, 1, :] = np_fill_t
    o3_t[0, 1, :] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    np_t = np_ref.copy()
    o3_t = o3c.Tensor(np_t, device=device)
    np_fill_t = np.random.rand(*np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3].shape)
    o3_fill_t = o3c.Tensor(np_fill_t, device=device)
    np_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = np_fill_t
    o3_t[0:2, 1:3, 0:4][0:1, 0:2, 2:3] = o3_fill_t
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    # Scalar boolean set item
    np_t = np.eye(4, dtype=np.bool8)
    o3_t = o3c.Tensor.eye(4, dtype=o3c.bool)
    np_t[2, 2] = False
    o3_t[2, 2] = False
    np.testing.assert_equal(o3_t.cpu().numpy(), np_t)

    # Slice a zero-dim tensor
    # match=".*Cannot slice a scalar (0-dim) tensor.*"
    with pytest.raises(RuntimeError,
                       match=r"Cannot slice a scalar \(0-dim\) tensor."):
        o3c.Tensor.ones((), device=device)[:] = 0
    with pytest.raises(RuntimeError,
                       match=r"Cannot slice a scalar \(0-dim\) tensor."):
        o3c.Tensor.ones((), device=device)[0:1] = 0


@pytest.mark.parametrize(
    "dim",
    [0, 1, 2, (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2), None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_sum(dim, keepdim, device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3c.Tensor(np_src, device=device)

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
    o3_src = o3c.Tensor(np_src, device=device)
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
    o3_src = o3c.Tensor(np_src, device=device)

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
    o3_src = o3c.Tensor(np_src, device=device)

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
    o3_src = o3c.Tensor(np_src, device=device)

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
    o3_src = o3c.Tensor(np_src, device=device)

    np_dst = np_src.max(axis=dim, keepdims=keepdim)
    o3_dst = o3_src.max(dim=dim, keepdim=keepdim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("dim", [0, 1, 2, None])
@pytest.mark.parametrize("device", list_devices())
def test_reduction_argmin_argmax(dim, device):
    np_src = np.array(range(24))
    np.random.shuffle(np_src)
    np_src = np_src.reshape((2, 3, 4))
    o3_src = o3c.Tensor(np_src, device=device)

    np_dst = np_src.argmin(axis=dim)
    o3_dst = o3_src.argmin(dim=dim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)

    np_dst = np_src.argmax(axis=dim)
    o3_dst = o3_src.argmax(dim=dim)
    np.testing.assert_allclose(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("device", list_devices())
def test_advanced_index_get_mixed(device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3c.Tensor(np_src, device=device)

    np_dst = np_src[1, 0:2, [1, 2]]
    o3_dst = o3_src[1, 0:2, [1, 2]]
    np.testing.assert_equal(o3_dst.cpu().numpy(), np_dst)

    # Subtle differences between slice and list
    np_src = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800]).reshape(3, 3)
    o3_src = o3c.Tensor(np_src, device=device)
    np.testing.assert_equal(o3_src[1, 2].cpu().numpy(), np_src[1, 2])
    np.testing.assert_equal(o3_src[[1, 2]].cpu().numpy(), np_src[[1, 2]])
    np.testing.assert_equal(o3_src[(1, 2)].cpu().numpy(), np_src[(1, 2)])
    np.testing.assert_equal(o3_src[(1, 2), [1, 2]].cpu().numpy(),
                            np_src[(1, 2), [1, 2]])

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3c.Tensor(np_src, device=device)
    o3_dst = o3_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np_dst = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]
    np.testing.assert_equal(o3_dst.cpu().numpy(), np_dst)


@pytest.mark.parametrize("device", list_devices())
def test_advanced_index_set_mixed(device):
    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3c.Tensor(np_src, device=device)

    np_fill = np.array(([[100, 200], [300, 400]]))
    o3_fill = o3c.Tensor(np_fill, device=device)

    np_src[1, 0:2, [1, 2]] = np_fill
    o3_src[1, 0:2, [1, 2]] = o3_fill
    np.testing.assert_equal(o3_src.cpu().numpy(), np_src)

    # Complex case: interleaving slice and advanced indexing
    np_src = np.array(range(120)).reshape((2, 3, 4, 5))
    o3_src = o3c.Tensor(np_src, device=device)
    fill_shape = np_src[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]].shape
    np_fill_val = np.random.randint(5000, size=fill_shape).astype(np_src.dtype)
    o3_fill_val = o3c.Tensor(np_fill_val)
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
    o3_t = o3c.Tensor(np_t, device=device)

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
    o3_a = o3c.Tensor(np_a, device=device)
    o3_b = o3c.Tensor(np_b, device=device)

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
    o3_a = o3c.Tensor(np_a, device=device)
    o3_b = o3c.Tensor(np_b, device=device)

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
    o3_x = o3c.Tensor(np_x, device=device)
    o3_nonzero_tuple = o3_x.nonzero(as_tuple=True)
    for np_t, o3_t in zip(np_nonzero_tuple, o3_nonzero_tuple):
        np.testing.assert_equal(np_t, o3_t.cpu().numpy())


@pytest.mark.parametrize("device", list_devices())
def test_boolean_advanced_indexing(device):
    np_a = np.array([1, -1, -2, 3])
    o3_a = o3c.Tensor(np_a, device=device)
    np_a[np_a < 0] = 0
    o3_a[o3_a < 0] = 0
    np.testing.assert_equal(np_a, o3_a.cpu().numpy())

    np_x = np.array([[0, 1], [1, 1], [2, 2]])
    np_row_sum = np.array([1, 2, 4])
    np_y = np_x[np_row_sum <= 2, :]
    o3_x = o3c.Tensor(np_x, device=device)
    o3_row_sum = o3c.Tensor(np_row_sum)
    o3_y = o3_x[o3_row_sum <= 2, :]
    np.testing.assert_equal(np_y, o3_y.cpu().numpy())

    np_t = np.array(5)
    np_t[np.array(True)] = 10
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(True, device=device)] = 10
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    np_t[np.array(True)] = np.array(10)
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(True, device=device)] = o3c.Tensor(10, device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    np_t[np.array(True)] = np.array([10])
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(True, device=device)] = o3c.Tensor([10], device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    with pytest.raises(Exception):
        np_t[np.array(True)] = np.array([[10]])
    o3_t = o3c.Tensor(5, device=device)
    with pytest.raises(Exception):
        o3_t[o3c.Tensor(True, device=device)] = o3c.Tensor([[10]],
                                                           device=device)

    np_t = np.array(5)
    with pytest.raises(Exception):
        np_t[np.array(True)] = np.array([10, 11])
    o3_t = o3c.Tensor(5, device=device)
    with pytest.raises(Exception):
        o3_t[o3c.Tensor(True, device=device)] = o3c.Tensor([10, 11],
                                                           device=device)

    np_t = np.array(5)
    np_t[np.array(False)] = 10
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(False, device=device)] = 10
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    np_t[np.array(False)] = np.array(10)
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(False, device=device)] = o3c.Tensor(10, device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    np_t[np.array(False)] = np.array([10])
    o3_t = o3c.Tensor(5, device=device)
    o3_t[o3c.Tensor(False, device=device)] = o3c.Tensor([10], device=device)
    np.testing.assert_equal(np_t, o3_t.cpu().numpy())

    np_t = np.array(5)
    with pytest.raises(Exception):
        np_t[np.array(False)] = np.array([[10]])
    o3_t = o3c.Tensor(5, device=device)
    with pytest.raises(Exception):
        o3_t[o3c.Tensor(False, device=device)] = o3c.Tensor([[10]],
                                                            device=device)

    np_t = np.array(5)
    with pytest.raises(Exception):
        np_t[np.array(False)] = np.array([10, 11])
    o3_t = o3c.Tensor(5, device=device)
    with pytest.raises(Exception):
        o3_t[o3c.Tensor(False, device=device)] = o3c.Tensor([10, 11],
                                                            device=device)


@pytest.mark.parametrize("device", list_devices())
def test_scalar_op(device):
    # +
    a = o3c.Tensor.ones((2, 3), o3c.float32, device=device)
    b = a.add(1)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = a + 1
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = 1 + a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))
    b = a + True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))

    # +=
    a = o3c.Tensor.ones((2, 3), o3c.float32, device=device)
    a.add_(1)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2))
    a += 1
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 3))
    a += True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 4))

    # -
    a = o3c.Tensor.ones((2, 3), o3c.float32, device=device)
    b = a.sub(1)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))
    b = a - 1
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))
    b = 10 - a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 9))
    b = a - True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 0))

    # -=
    a = o3c.Tensor.ones((2, 3), o3c.float32, device=device)
    a.sub_(1)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 0))
    a -= 1
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), -1))
    a -= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), -2))

    # *
    a = o3c.Tensor.full((2, 3), 2, o3c.float32, device=device)
    b = a.mul(10)
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = a * 10
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = 10 * a
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 20))
    b = a * True
    np.testing.assert_equal(b.cpu().numpy(), np.full((2, 3), 2))

    # *=
    a = o3c.Tensor.full((2, 3), 2, o3c.float32, device=device)
    a.mul_(10)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 20))
    a *= 10
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 200))
    a *= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 200))

    # /
    a = o3c.Tensor.full((2, 3), 20, o3c.float32, device=device)
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
    a = o3c.Tensor.full((2, 3), 20, o3c.float32, device=device)
    a.div_(2)
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 10))
    a /= 2
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 5))
    a //= 2
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2.5))
    a /= True
    np.testing.assert_equal(a.cpu().numpy(), np.full((2, 3), 2.5))

    # logical_and
    a = o3c.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_and(True).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_and(5).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_and(False).cpu().numpy(), np.array([False, False]))
    np.testing.assert_equal(
        a.logical_and(0).cpu().numpy(), np.array([False, False]))

    # logical_and_
    a = o3c.Tensor([True, False], device=device)
    a.logical_and_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_and_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_and_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False]))
    a.logical_and_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False]))

    # logical_or
    a = o3c.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_or(True).cpu().numpy(), np.array([True, True]))
    np.testing.assert_equal(
        a.logical_or(5).cpu().numpy(), np.array([True, True]))
    np.testing.assert_equal(
        a.logical_or(False).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_or(0).cpu().numpy(), np.array([True, False]))

    # logical_or_
    a = o3c.Tensor([True, False], device=device)
    a.logical_or_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_or_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_or_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a.logical_or_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))

    # logical_xor
    a = o3c.Tensor([True, False], device=device)
    np.testing.assert_equal(
        a.logical_xor(True).cpu().numpy(), np.array([False, True]))
    np.testing.assert_equal(
        a.logical_xor(5).cpu().numpy(), np.array([False, True]))
    np.testing.assert_equal(
        a.logical_xor(False).cpu().numpy(), np.array([True, False]))
    np.testing.assert_equal(
        a.logical_xor(0).cpu().numpy(), np.array([True, False]))

    # logical_xor_
    a = o3c.Tensor([True, False], device=device)
    a.logical_xor_(True)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_xor_(5)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True]))
    a = o3c.Tensor([True, False], device=device)
    a.logical_xor_(False)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))
    a.logical_xor_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False]))

    # gt
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.gt(0)).cpu().numpy(),
                            np.array([False, False, True]))
    np.testing.assert_equal((a > 0).cpu().numpy(),
                            np.array([False, False, True]))

    # gt_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.gt_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, False, True]))

    # lt
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.lt(0)).cpu().numpy(),
                            np.array([True, False, False]))
    np.testing.assert_equal((a < 0).cpu().numpy(),
                            np.array([True, False, False]))

    # lt_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.lt_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False, False]))

    # ge
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.ge(0)).cpu().numpy(),
                            np.array([False, True, True]))
    np.testing.assert_equal((a >= 0).cpu().numpy(),
                            np.array([False, True, True]))

    # ge_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.ge_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True, True]))

    # le
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.le(0)).cpu().numpy(),
                            np.array([True, True, False]))
    np.testing.assert_equal((a <= 0).cpu().numpy(),
                            np.array([True, True, False]))

    # le_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.le_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, True, False]))

    # eq
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.eq(0)).cpu().numpy(),
                            np.array([False, True, False]))
    np.testing.assert_equal((a == 0).cpu().numpy(),
                            np.array([False, True, False]))

    # eq_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.eq_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([False, True, False]))

    # ne
    dtype = o3c.float32
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    np.testing.assert_equal((a.ne(0)).cpu().numpy(),
                            np.array([True, False, True]))
    np.testing.assert_equal((a != 0).cpu().numpy(),
                            np.array([True, False, True]))

    # ne_
    a = o3c.Tensor([-1, 0, 1], dtype=dtype, device=device)
    a.ne_(0)
    np.testing.assert_equal(a.cpu().numpy(), np.array([True, False, True]))

    # clip
    dtype = o3c.int64
    a = o3c.Tensor([2, -1, 1], dtype=dtype, device=device)
    np.testing.assert_equal(a.clip(0, 1).cpu().numpy(), np.array([1, 0, 1]))
    np.testing.assert_equal(a.clip(0.5, 1.2).cpu().numpy(), np.array([1, 0, 1]))

    # clip_
    a = o3c.Tensor([2, -1, 1], dtype=dtype, device=device)
    a.clip_(0, 1)
    np.testing.assert_equal(a.cpu().numpy(), np.array([1, 0, 1]))


@pytest.mark.parametrize("device", list_devices())
def test_all_any(device):
    a = o3c.Tensor([False, True, True, True], dtype=o3c.bool, device=device)
    assert not a.all()
    assert a.any()

    a = o3c.Tensor([True, True, True, True], dtype=o3c.bool, device=device)
    assert a.all()

    # Empty
    a = o3c.Tensor([], dtype=o3c.bool, device=device)
    assert a.all()
    assert not a.any()


@pytest.mark.parametrize("device", list_devices())
def test_allclose_isclose(device):
    a = o3c.Tensor([1, 2], device=device)
    b = o3c.Tensor([1, 3], device=device)
    assert not a.allclose(b)
    np.testing.assert_allclose(
        a.isclose(b).cpu().numpy(), np.array([True, False]))

    assert a.allclose(b, atol=1)
    np.testing.assert_allclose(
        a.isclose(b, atol=1).cpu().numpy(), np.array([True, True]))

    # Test cases from
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    a = o3c.Tensor([1e10, 1e-7], device=device)
    b = o3c.Tensor([1.00001e10, 1e-8], device=device)
    assert not a.allclose(b)
    a = o3c.Tensor([1e10, 1e-8], device=device)
    b = o3c.Tensor([1.00001e10, 1e-9], device=device)
    assert a.allclose(b)
    a = o3c.Tensor([1e10, 1e-8], device=device)
    b = o3c.Tensor([1.0001e10, 1e-9], device=device)
    assert not a.allclose(b)


@pytest.mark.parametrize("device", list_devices())
def test_issame(device):
    dtype = o3c.float32
    a = o3c.Tensor.ones((2, 3), dtype, device=device)
    b = o3c.Tensor.ones((2, 3), dtype, device=device)
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
    o3_t = o3c.Tensor.ones((2, 3), dtype=o3c.float32, device=device) * 1.5
    assert o3_t[0, 0].item() == 1.5
    assert isinstance(o3_t[0, 0].item(), float)

    o3_t = o3c.Tensor.ones((2, 3), dtype=o3c.float64, device=device) * 1.5
    assert o3_t[0, 0].item() == 1.5
    assert isinstance(o3_t[0, 0].item(), float)

    o3_t = o3c.Tensor.ones((2, 3), dtype=o3c.int32, device=device) * 1.5
    assert o3_t[0, 0].item() == 1
    assert isinstance(o3_t[0, 0].item(), int)

    o3_t = o3c.Tensor.ones((2, 3), dtype=o3c.int64, device=device) * 1.5
    assert o3_t[0, 0].item() == 1
    assert isinstance(o3_t[0, 0].item(), int)

    o3_t = o3c.Tensor.ones((2, 3), dtype=o3c.bool, device=device)
    assert o3_t[0, 0].item() == True
    assert isinstance(o3_t[0, 0].item(), bool)


@pytest.mark.parametrize("device", list_devices())
def test_save_load(device):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/tensor.npy"

        o3_tensors = [
            o3c.Tensor([[1, 2], [3, 4]], dtype=o3c.float32, device=device),
            o3c.Tensor(3.14, dtype=o3c.float32, device=device),
            o3c.Tensor.ones((0,), dtype=o3c.float32, device=device),
            o3c.Tensor.ones((0, 0), dtype=o3c.float32, device=device),
            o3c.Tensor.ones((0, 1, 0), dtype=o3c.float32, device=device)
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
            o3_t_load = o3c.Tensor.load(file_name)
            np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)

            # Open3D -> Numpy.
            np_t_load = np.load(file_name)
            np.testing.assert_equal(np_t_load, np_t_load)

            # Numpy -> Open3D.
            np.save(file_name, np_t)
            o3_t_load = o3c.Tensor.load(file_name)
            np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)

        # Ragged tensor: exception.
        np_t = np.array([[1, 2, 3], [4, 5]], dtype=np.dtype(object))
        np.save(file_name, np_t)
        with pytest.raises(Exception):
            o3_t_load = o3c.Tensor.load(file_name)

        # Fortran order: exception.
        np_t = np.array([[1, 2, 3], [4, 5, 6]])
        np_t = np.asfortranarray(np_t)
        np.save(file_name, np_t)
        with pytest.raises(Exception):
            o3_t_load = o3c.Tensor.load(file_name)

        # Unsupported dtype: exception.
        np_t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
        np.save(file_name, np_t)
        with pytest.raises(Exception):
            o3_t_load = o3c.Tensor.load(file_name)

        # Non-contiguous numpy array.
        np_t = np.arange(24).reshape(2, 3, 4)
        assert np_t.flags['C_CONTIGUOUS']
        np_t = np_t[0:2:1, 0:3:2, 0:4:2]
        assert not np_t.flags['C_CONTIGUOUS']
        np.save(file_name, np_t)
        o3_t_load = o3c.Tensor.load(file_name)
        assert o3_t_load.is_contiguous()
        np.testing.assert_equal(o3_t_load.cpu().numpy(), np_t)


@pytest.mark.parametrize("device", list_devices())
def test_iterator(device):
    # 0-d.
    o3_t = o3c.Tensor.ones((), dtype=o3c.float32, device=device)
    with pytest.raises(Exception, match=r'Cannot iterate a scalar'):
        for o3_t_slice in o3_t:
            pass

    # 1-d.
    o3_t = o3c.Tensor([0, 1, 2], device=device)
    np_t = np.array([0, 1, 2])
    for o3_t_slice, np_t_slice in zip(o3_t, np_t):
        np.testing.assert_equal(o3_t_slice.cpu().numpy(), np_t_slice)

    # TODO: 1-d with assignment (assigning to a 0-d slice) is not possible with
    # our current slice API. This issue is not related to the iterator.
    # Operator [:] requires 1 or more dimensions. We need to implement the [...]
    # operator. See: https://stackoverflow.com/a/49581266/1255535.

    # 2-d.
    o3_t = o3c.Tensor([[0, 1, 2], [3, 4, 5]], device=device)
    np_t = np.array([[0, 1, 2], [3, 4, 5]])
    for o3_t_slice, np_t_slice in zip(o3_t, np_t):
        np.testing.assert_equal(o3_t_slice.cpu().numpy(), np_t_slice)

    # 2-d with assignment.
    o3_t = o3c.Tensor([[0, 1, 2], [3, 4, 5]], device=device)
    new_o3_t_slices = [
        o3c.Tensor([0, 10, 20], device=device),
        o3c.Tensor([30, 40, 50], device=device)
    ]
    for o3_t_slice, new_o3_t_slice in zip(o3_t, new_o3_t_slices):
        o3_t_slice[:] = new_o3_t_slice
    np.testing.assert_equal(o3_t.cpu().numpy(),
                            np.array([[0, 10, 20], [30, 40, 50]]))
