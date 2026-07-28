# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest
import mltest

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


@mltest.parametrize.ml_gpu_only
def test_three_nn(ml):

    values0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/values0.npy'
    )
    values1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/values1.npy'
    )

    ans0, ans1 = mltest.run_op(ml, ml.device, True, ml.ops.three_nn, values0,
                               values1)

    expected0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/out0.npy'
    )
    expected1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/three_nn/out1.npy'
    )
    # three_nn's SYCL implementation finds the top-3 via a work-group
    # cooperative parallel scan rather than a single serial scan (see
    # ThreeNNSYCL in InterpolatePointsSYCL.h), so when candidates are
    # equidistant the specific index chosen (and the order among the 3
    # slots) can differ from the CPU/CUDA reference while still being a
    # valid nearest-3, so compare each query's *set* of (distance, index)
    # pairs rather than the exact slot order. ans0/expected0 are squared
    # distances (float); round before hashing to tolerate floating-point
    # noise from the different summation order.
    ans0_r = np.round(ans0, decimals=4)
    expected0_r = np.round(expected0, decimals=4)
    ans_sets = [
        set(zip(d_row, i_row))
        for d_row, i_row in zip(ans0_r.reshape(-1, 3), ans1.reshape(-1, 3))
    ]
    expected_sets = [
        set(zip(d_row, i_row)) for d_row, i_row in zip(
            expected0_r.reshape(-1, 3), expected1.reshape(-1, 3))
    ]
    assert ans_sets == expected_sets
