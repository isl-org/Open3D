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
def test_roi_pool(ml):

    values0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/roi_pool/values0.npy'
    )
    values1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/roi_pool/values1.npy'
    )
    values2 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/roi_pool/values2.npy'
    )
    sampled_pts_num = 512

    ans0, ans1 = mltest.run_op(ml, ml.device, True, ml.ops.roi_pool, values0,
                               values1, values2, sampled_pts_num)

    expected0 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/roi_pool/out0.npy'
    )
    expected1 = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/roi_pool/out1.npy'
    )

    np.testing.assert_equal(ans0, expected0)
    np.testing.assert_equal(ans1, expected1)
