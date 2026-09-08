# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest
import mltest

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


@mltest.parametrize.ml_gpu_and_torch_cpu
def test_three_interp(ml):

    values0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values0.npy'
    )
    values1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values1.npy'
    )
    values2 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/values2.npy'
    )

    ans = mltest.run_op(ml, ml.device, True, ml.ops.three_interpolate, values0,
                        values1, values2)

    expected = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_interp/out.npy'
    )
    # The CPU kernel sums the three weighted terms in a different order (and
    # without the fused-multiply-add pattern used by the CUDA/SYCL kernels),
    # so results can differ by a few float32 ULPs; use a tight numerical
    # tolerance instead of exact equality.
    np.testing.assert_allclose(ans, expected, rtol=1e-5, atol=1e-6)


@mltest.parametrize.ml_gpu_and_torch_cpu
def test_three_interp_grad(ml):
    rng = np.random.RandomState(0)
    b, c, n, m = 2, 4, 20, 10

    idx = rng.randint(0, m, size=(b, n, 3)).astype(np.int32)
    weights = rng.uniform(0, 1, size=(b, n, 3)).astype(np.float32)
    grad_out = rng.uniform(-1, 1, size=(b, c, n)).astype(np.float32)

    grad_x = mltest.run_op(ml, ml.device, True, ml.ops.three_interpolate_grad,
                           grad_out, idx, weights, m)

    assert grad_x.shape == (b, c, m)

    # Reference: scatter-add matching three_interpolate's forward mapping
    # out[b,c,n] = sum_k weights[b,n,k] * points[b,c,idx[b,n,k]], so its
    # transpose scatters grad_out back through the same (idx, weights) pairs.
    expected = np.zeros((b, c, m), dtype=np.float32)
    for bi in range(b):
        for k in range(3):
            np.add.at(expected[bi], (slice(None), idx[bi, :, k]),
                      weights[bi, :, k][None, :] * grad_out[bi])

    np.testing.assert_allclose(grad_x, expected, rtol=1e-4, atol=1e-4)
