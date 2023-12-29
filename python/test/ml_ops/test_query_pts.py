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
def test_query_pts(ml):

    values0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/query_pts/values0.npy'
    )
    values1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/query_pts/values1.npy'
    )

    sample = 16
    radius = 0.1

    ans = mltest.run_op(ml, ml.device, True, ml.ops.ball_query, values0,
                        values1, radius, sample)

    expected = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/query_pts/out.npy'
    )
    np.testing.assert_equal(ans, expected)
