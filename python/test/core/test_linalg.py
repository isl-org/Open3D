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
import core_test_utils

if core_test_utils.torch_available():
    import torch
    import torch.utils.dlpack


@pytest.mark.parametrize("device", core_test_utils.list_devices())
@pytest.mark.parametrize("dtype", [
    o3d.core.Dtype.Int32, o3d.core.Dtype.Int64, o3d.core.Dtype.Float32,
    o3d.core.Dtype.Float64
])
def test_matmul(device, dtype):
    # Shape takes tuple, list or o3d.core.SizeVector
    a = o3d.core.Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    b = o3d.core.Tensor([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]],
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


@pytest.mark.parametrize("device", core_test_utils.list_devices())
@pytest.mark.parametrize("dtype",
                         [o3d.core.Dtype.Float32, o3d.core.Dtype.Float64])
def test_inverse(device, dtype):
    a = o3d.core.Tensor([[7, 2, 1], [0, 3, -1], [-3, 4, 2]],
                        dtype=dtype,
                        device=device)
    a_inv = o3d.core.inv(a)
    a_inv_numpy = np.linalg.inv(a.cpu().numpy())
    np.testing.assert_allclose(a_inv.cpu().numpy(), a_inv_numpy, 1e-6)

    # Singular condition
    a = o3d.core.Tensor([[0, 0], [0, 1]], dtype=dtype, device=device)
    try:
        a_inv = a.inv()
    except RuntimeError:
        pass
    else:  # should never reach here
        assert False


@pytest.mark.parametrize("device", core_test_utils.list_devices())
@pytest.mark.parametrize("dtype",
                         [o3d.core.Dtype.Float32, o3d.core.Dtype.Float64])
def test_svd(device, dtype):
    a = o3d.core.Tensor([[2, 4], [1, 3], [0, 0], [0, 0]],
                        dtype=dtype,
                        device=device)
    u, s, vt = o3d.core.svd(a)
    assert u.shape == o3d.core.SizeVector([4, 4])
    assert s.shape == o3d.core.SizeVector([2])
    assert vt.shape == o3d.core.SizeVector([2, 2])

    # u and vt are orthogonal matrices
    uut = u @ u.transpose()
    eye_uut = o3d.core.Tensor.eye(4, dtype=dtype)
    np.testing.assert_allclose(uut.cpu().numpy(),
                               eye_uut.cpu().numpy(),
                               atol=1e-6)

    vvt = vt.transpose() @ vt
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


@pytest.mark.parametrize("device", core_test_utils.list_devices())
@pytest.mark.parametrize("dtype",
                         [o3d.core.Dtype.Float32, o3d.core.Dtype.Float64])
def test_solve(device, dtype):
    # Test square
    a = o3d.core.Tensor([[3, 1], [1, 2]], dtype=dtype, device=device)
    b = o3d.core.Tensor([9, 8], dtype=dtype, device=device)
    x = o3d.core.solve(a, b)

    x_numpy = np.linalg.solve(a.cpu().numpy(), b.cpu().numpy())
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
    x = a.solve(b)
    x_numpy, _, _, _ = np.linalg.lstsq(a.cpu().numpy(),
                                       b.cpu().numpy(),
                                       rcond=None)
    np.testing.assert_allclose(x.cpu().numpy(), x_numpy, atol=1e-6)
