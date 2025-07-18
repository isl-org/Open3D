# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest
import importlib


@pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                    reason='tf ops not built')
def test_tf_subsampling():
    ops = importlib.import_module('open3d.ml.tf.ops')

    points = np.array(range(21)).reshape(-1, 3).astype(np.float32)

    sub_points = ops.grid_subsampling(points, 10).cpu().numpy()
    sub_points_ref = np.array(
        [[13.5, 14.5, 15.5], [18, 19, 20], [9, 10, 11], [3, 4, 5]],
        dtype=np.float32)

    sub_points = sub_points[sub_points[:, 0].argsort()]
    sub_points_ref = sub_points_ref[sub_points_ref[:, 0].argsort()]
    np.testing.assert_equal(sub_points, sub_points_ref)

    sub_points = ops.grid_subsampling(points, 12).cpu().numpy()
    sub_points_ref = np.array([[15, 16, 17], [4.5, 5.5, 6.5]], dtype=np.float32)
    sub_points = sub_points[sub_points[:, 0].argsort()]
    sub_points_ref = sub_points_ref[sub_points_ref[:, 0].argsort()]

    np.testing.assert_equal(sub_points, sub_points_ref)

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    sub_points_ref = np.array([[5, 0.5, 0], [0.4, 0.4, 0.4]], dtype=np.float32)
    sub_points = ops.grid_subsampling(points, 1.1).cpu().numpy()
    sub_points = sub_points[sub_points[:, 0].argsort()]
    sub_points_ref = sub_points_ref[sub_points_ref[:, 0].argsort()]

    np.testing.assert_equal(sub_points, sub_points_ref)

    with pytest.raises(ValueError):
        ops.grid_subsampling(None, 1)


@pytest.mark.skipif(not o3d._build_config['BUILD_TENSORFLOW_OPS'],
                    reason='tf ops not built')
def test_tf_batch_subsampling():
    ops = importlib.import_module('open3d.ml.tf.ops')

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    batches = np.array([3, 2, 2], dtype=np.int32)
    sub_points_ref = np.array(
        [[0.3333333, 0.3333333, 0], [0.5, 0.5, 1], [5, 0.5, 0]],
        dtype=np.float32)
    sub_batch_ref = np.array([1, 1, 1], dtype=np.int32)

    (sub_points, sub_batch) = ops.batch_grid_subsampling(points, batches, 1.1)
    sub_points, sub_batch = sub_points.cpu().numpy(), sub_batch.cpu().numpy()

    np.testing.assert_almost_equal(sub_points, sub_points_ref)
    np.testing.assert_almost_equal(sub_batch, sub_batch_ref)

    with pytest.raises(ValueError):
        ops.batch_grid_subsampling(None, batches, 1)

    with pytest.raises(ValueError):
        ops.batch_grid_subsampling(points, None, 1)
