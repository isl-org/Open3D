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


@mltest.parametrize.ml
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
    # Furthest point sampling picks the farthest-away candidate at each step
    # via a max-reduction; when two candidates are float32-exact-tied for
    # farthest, the CPU kernel's serial reduction and the CUDA kernel's
    # parallel-tree reduction can validly pick either one first. This only
    # swaps *when* each tied point gets selected, not *whether* it is
    # selected, so compare the selected index sets rather than the exact
    # per-step order.
    np.testing.assert_equal(np.sort(ans, axis=-1), np.sort(expected, axis=-1))
