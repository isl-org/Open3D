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
def test_furthest_point_sampling(ml):

    values = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/sampling/values.npy'
    )
    samples = 4096

    ans = mltest.run_op(ml, ml.device, True, ml.ops.furthest_point_sampling,
                        values, samples)

    expected = mltest.fetch_numpy(
        'https://storage.googleapis.com/isl-datasets/open3d-dev/test/ml_ops/data/sampling/out.npy'
    )
    np.testing.assert_equal(ans, expected)
