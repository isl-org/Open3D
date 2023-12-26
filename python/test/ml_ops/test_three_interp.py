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

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


@mltest.parametrize.ml_gpu_only
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
    np.testing.assert_equal(ans, expected)
