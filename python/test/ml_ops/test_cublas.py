# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest
import mltest

# Skip all tests if the ml ops were not built
pytestmark = mltest.default_marks


@mltest.parametrize.ml_gpu_only
def test_cublas_matmul(ml):
    # This test checks if calling cublas functionality from open3d and the ml framework works.

    rng = np.random.RandomState(123)

    n = 20
    arr = rng.rand(n, n).astype(np.float32)

    # do matmul with open3d
    A = o3d.core.Tensor.from_numpy(arr).cuda()
    B = A @ A

    # now use the ml framework cublas
    C = mltest.run_op(ml, ml.device, True, ml.module.matmul, arr, arr)

    np.testing.assert_allclose(B.cpu().numpy(), C)
