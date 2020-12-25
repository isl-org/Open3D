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
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype", [
    o3d.core.Dtype.Int32, o3d.core.Dtype.Int64, o3d.core.Dtype.Float32,
    o3d.core.Dtype.Float64
])
def test_matmul(device, dtype):
    # Shape takes tuple, list or o3d.core.SizeVector
    a = o3d.core.Tensor([[1, 2.5, 3], [4, 5, 6.2]], dtype=dtype, device=device)
    b = o3d.core.Tensor([[7.5, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17.8, 18]],
                        dtype=dtype,
                        device=device)
    c = o3d.core.matmul(a, b)
    assert c.shape == o3d.core.SizeVector([2, 4])

    c_numpy = a.cpu().numpy() @ b.cpu().numpy()
    np.testing.assert_allclose(c.cpu().numpy(), c_numpy, 1e-6)

    # Non-contiguous test
    a = a[:, 1:]
    b = b[[0, 2], :]
    c = a.matmul(b)
    assert c.shape == o3d.core.SizeVector([2, 4])

    c_numpy = a.cpu().numpy() @ b.cpu().numpy()
    np.testing.assert_allclose(c.cpu().numpy(), c_numpy, 1e-6)

    # Incompatible shape test
    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor.zeros((3, 4, 5), dtype=dtype)
        b = o3d.core.Tensor.zeros((4, 5), dtype=dtype)
        c = a @ b
    assert 'Tensor A must be 2D' in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor.zeros((3, 4), dtype=dtype)
        b = o3d.core.Tensor.zeros((4, 5, 6), dtype=dtype)
        c = a @ b
    assert 'Tensor B must be 1D (vector) or 2D (matrix)' in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor.zeros((3, 4), dtype=dtype)
        b = o3d.core.Tensor.zeros((3, 7), dtype=dtype)
        c = a @ b
    assert 'mismatch with' in str(excinfo.value)

    for shapes in [((0, 0), (0, 0)), ((2, 0), (0, 3)), ((0, 2), (2, 0)),
                   ((2, 0), (0, 0))]:
        with pytest.raises(RuntimeError) as excinfo:
            a_shape, b_shape = shapes
            a = o3d.core.Tensor.zeros(a_shape, dtype=dtype, device=device)
            b = o3d.core.Tensor.zeros(b_shape, dtype=dtype, device=device)
            c = a @ b
        assert 'dimensions with zero' in str(excinfo.value)


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype", [
    o3d.core.Dtype.Int32, o3d.core.Dtype.Int64, o3d.core.Dtype.Float32,
    o3d.core.Dtype.Float64
])
def test_inverse(device, dtype):
    a = o3d.core.Tensor([[7, 2, 1], [0, 3, -1], [-3, 4, 2]],
                        dtype=dtype,
                        device=device)

    if dtype in [o3d.core.Dtype.Int32, o3d.core.Dtype.Int64]:
        with pytest.raises(RuntimeError) as excinfo:
            o3d.core.inv(a)
        assert 'Only tensors with Float32 or Float64 are supported' in str(
            excinfo.value)
        return

    a_inv = o3d.core.inv(a)
    a_inv_numpy = np.linalg.inv(a.cpu().numpy())
    np.testing.assert_allclose(a_inv.cpu().numpy(),
                               a_inv_numpy,
                               rtol=1e-5,
                               atol=1e-5)

    # Non-2D
    for shape in [(), [1], (3, 4, 5)]:
        with pytest.raises(RuntimeError) as excinfo:
            a = o3d.core.Tensor.zeros(shape, dtype=dtype, device=device)
            a.inv()
        assert 'must be 2D' in str(excinfo.value)

    # Non-square
    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor.zeros((2, 3), dtype=dtype, device=device)
        a.inv()
    assert 'must be square' in str(excinfo.value)

    a = o3d.core.Tensor([[1]], dtype=dtype, device=device)
    np.testing.assert_allclose(a.cpu().numpy(), a.inv().cpu().numpy(), 1e-6)

    # Singular condition
    for a in [
            o3d.core.Tensor([[0, 0], [0, 1]], dtype=dtype, device=device),
            o3d.core.Tensor([[0]], dtype=dtype, device=device)
    ]:
        with pytest.raises(RuntimeError) as excinfo:
            a_inv = a.inv()
        assert 'singular condition' in str(excinfo.value)

        with pytest.raises(np.linalg.LinAlgError) as excinfo:
            a_inv = np.linalg.inv(a.cpu().numpy())
        assert 'Singular matrix' in str(excinfo.value)


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype", [
    o3d.core.Dtype.Int32, o3d.core.Dtype.Int64, o3d.core.Dtype.Float32,
    o3d.core.Dtype.Float64
])
def test_svd(device, dtype):
    a = o3d.core.Tensor([[2, 4], [1, 3], [0, 0], [0, 0]],
                        dtype=dtype,
                        device=device)
    if dtype in [o3d.core.Dtype.Int32, o3d.core.Dtype.Int64]:
        with pytest.raises(RuntimeError) as excinfo:
            o3d.core.svd(a)
        assert 'Only tensors with Float32 or Float64 are supported' in str(
            excinfo.value)
        return

    u, s, vt = o3d.core.svd(a)
    assert u.shape == o3d.core.SizeVector([4, 4])
    assert s.shape == o3d.core.SizeVector([2])
    assert vt.shape == o3d.core.SizeVector([2, 2])

    # u and vt are orthogonal matrices
    uut = u @ u.T()
    eye_uut = o3d.core.Tensor.eye(4, dtype=dtype)
    np.testing.assert_allclose(uut.cpu().numpy(),
                               eye_uut.cpu().numpy(),
                               atol=1e-6)

    vvt = vt.T() @ vt
    eye_vvt = o3d.core.Tensor.eye(2, dtype=dtype)
    np.testing.assert_allclose(vvt.cpu().numpy(),
                               eye_vvt.cpu().numpy(),
                               atol=1e-6)

    usvt = u[:, :2] @ o3d.core.Tensor.diag(s) @ vt
    # The accuracy of svd over Float32 is very low...
    np.testing.assert_allclose(a.cpu().numpy(), usvt.cpu().numpy(), atol=1e-5)

    u = u.cpu().numpy()
    s = s.cpu().numpy()
    vt = vt.cpu().numpy()
    u_numpy, s_numpy, vt_numpy = np.linalg.svd(a.cpu().numpy())

    # u, vt can be different by several signs
    np.testing.assert_allclose(np.abs(u), np.abs(u_numpy), 1e-6)
    np.testing.assert_allclose(np.abs(vt), np.abs(vt_numpy), 1e-6)

    np.testing.assert_allclose(s, s_numpy, 1e-6)

    for shapes in [(0, 0), (0, 2)]:
        with pytest.raises(RuntimeError) as excinfo:
            a = o3d.core.Tensor.zeros(shapes, dtype=dtype, device=device)
            a.svd()
        assert 'dimensions with zero' in str(excinfo.value)
    for shapes in [(), [1], (1, 2, 4)]:
        with pytest.raises(RuntimeError) as excinfo:
            a = o3d.core.Tensor.zeros(shapes, dtype=dtype, device=device)
            a.svd()
        assert 'must be 2D' in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor(shapes, dtype=dtype, device=device)
        a.svd()
    assert 'must be 2D' in str(excinfo.value)


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype",
                         [o3d.core.Dtype.Float32, o3d.core.Dtype.Float64])
def test_solve(device, dtype):
    # Test square
    a = o3d.core.Tensor([[3, 1], [1, 2]], dtype=dtype, device=device)
    b = o3d.core.Tensor([9, 8], dtype=dtype, device=device)
    x = o3d.core.solve(a, b)

    x_numpy = np.linalg.solve(a.cpu().numpy(), b.cpu().numpy())
    np.testing.assert_allclose(x.cpu().numpy(), x_numpy, atol=1e-6)

    with pytest.raises(RuntimeError) as excinfo:
        a = o3d.core.Tensor.zeros((3, 3), dtype=dtype, device=device)
        b = o3d.core.Tensor.ones((3,), dtype=dtype, device=device)
        x = o3d.core.solve(a, b)
    assert 'singular' in str(excinfo.value)


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype",
                         [o3d.core.Dtype.Float32, o3d.core.Dtype.Float64])
def test_lstsq(device, dtype):
    # Test square
    a = o3d.core.Tensor([[3, 1], [1, 2]], dtype=dtype, device=device)
    b = o3d.core.Tensor([9, 8], dtype=dtype, device=device)
    x = o3d.core.lstsq(a, b)

    x_numpy, _, _, _ = np.linalg.lstsq(a.cpu().numpy(),
                                       b.cpu().numpy(),
                                       rcond=None)
    np.testing.assert_allclose(x.cpu().numpy(), x_numpy, atol=1e-6)

    # Test non-square
    a = o3d.core.Tensor(
        [[1.44, -7.84, -4.39, 4.53], [-9.96, -0.28, -3.24, 3.83],
         [-7.55, 3.24, 6.27, -6.64], [8.34, 8.09, 5.28, 2.06],
         [7.08, 2.52, 0.74, -2.47], [-5.45, -5.70, -1.19, 4.70]],
        dtype=dtype,
        device=device)
    b = o3d.core.Tensor([[8.58, 9.35], [8.26, -4.43], [8.48, -0.70],
                         [-5.28, -0.26], [5.72, -7.36], [8.93, -2.52]],
                        dtype=dtype,
                        device=device)
    x = a.lstsq(b)
    x_numpy, _, _, _ = np.linalg.lstsq(a.cpu().numpy(),
                                       b.cpu().numpy(),
                                       rcond=None)
    np.testing.assert_allclose(x.cpu().numpy(), x_numpy, atol=1e-6)

    for shapes in [((0, 0), (0, 0)), ((2, 0), (2, 3)), ((2, 0), (2, 0))]:
        with pytest.raises(RuntimeError) as excinfo:
            a_shape, b_shape = shapes
            a = o3d.core.Tensor.zeros(a_shape, dtype=dtype, device=device)
            b = o3d.core.Tensor.zeros(b_shape, dtype=dtype, device=device)
            a.lstsq(b)
        assert 'dimensions with zero' in str(excinfo.value)

    for shapes in [((2, 3), (2, 2))]:
        with pytest.raises(RuntimeError) as excinfo:
            a_shape, b_shape = shapes
            a = o3d.core.Tensor.zeros(a_shape, dtype=dtype, device=device)
            b = o3d.core.Tensor.zeros(b_shape, dtype=dtype, device=device)
            a.lstsq(b)
        assert 'must satisfy rows({}) > cols({})'.format(
            a_shape[0], a_shape[1]) in str(excinfo.value)
