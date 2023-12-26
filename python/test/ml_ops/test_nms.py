# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import mltest

# Skip all tests if the ml ops were not built.
pytestmark = mltest.default_marks


@mltest.parametrize.ml
def test_nms(ml):
    boxes = np.array([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                      [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                      [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                      [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                      [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                      [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                     dtype=np.float32)
    scores = np.array([3, 1.1, 5, 2, 1, 0], dtype=np.float32)
    nms_overlap_thresh = 0.7
    keep_indices_ref = np.array([2, 3, 5]).astype(np.int64)

    keep_indices = mltest.run_op(ml,
                                 ml.device,
                                 True,
                                 ml.ops.nms,
                                 boxes,
                                 scores,
                                 nms_overlap_thresh=nms_overlap_thresh)

    np.testing.assert_equal(keep_indices, keep_indices_ref)
    assert keep_indices.dtype == keep_indices_ref.dtype


@mltest.parametrize.ml
def test_nms_empty(ml):
    boxes = np.zeros((0, 5), dtype=np.float32)
    scores = np.array([], dtype=np.float32)
    nms_overlap_thresh = 0.7
    keep_indices_ref = np.array([]).astype(np.int64)

    keep_indices = mltest.run_op(ml,
                                 ml.device,
                                 True,
                                 ml.ops.nms,
                                 boxes,
                                 scores,
                                 nms_overlap_thresh=nms_overlap_thresh)

    np.testing.assert_equal(keep_indices, keep_indices_ref)
    assert keep_indices.dtype == keep_indices_ref.dtype
