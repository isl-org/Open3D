# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
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
def test_nms_multiblock(ml):
    # NMS_BLOCK_SIZE (cpp/open3d/ml/contrib/IoUImpl.h) is 64: build >64 boxes
    # so the SYCL/CUDA kernel's bitmask exercises more than one block both
    # row- and column-wise.
    num_clusters = 20
    boxes_per_cluster = 5
    nms_overlap_thresh = 0.3

    boxes = []
    scores = []
    cluster_of = []
    score = 0.0
    for cluster_idx in range(num_clusters):
        # All boxes in a cluster share identical coordinates (angle=0), so
        # every pair within the cluster has IoU == 1 regardless of the exact
        # rotated-polygon overlap implementation; clusters are spaced far
        # enough apart (offset 4, box size 2) that no cross-cluster overlap
        # ever occurs. Keep coordinates well below ~200: CheckInBox2D's
        # absolute MARGIN (IoUImpl.h) loses precision in float32 once box
        # coordinates get much larger than that, which would spuriously
        # break same-cluster IoU==1 detection unrelated to what this test
        # is meant to exercise (the multi-block bitmask path).
        x0 = cluster_idx * 4.0
        for _ in range(boxes_per_cluster):
            boxes.append([x0, 0.0, x0 + 2.0, 2.0, 0.0])
            score += 1.0
            scores.append(score)
            cluster_of.append(cluster_idx)
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    cluster_of = np.array(cluster_of)
    n = boxes.shape[0]
    assert n > 64

    # Reference: NMS processes boxes in score-descending order and greedily
    # suppresses same-cluster boxes (IoU == 1); since scores strictly
    # increase with cluster_idx*boxes_per_cluster+j above, the highest-score
    # box in each cluster is exactly the last one added to that cluster.
    keep_per_cluster = [
        np.where(cluster_of == c)[0][-1] for c in range(num_clusters)
    ]
    keep_indices_ref = np.array(sorted(keep_per_cluster,
                                       key=lambda i: -scores[i]),
                                dtype=np.int64)

    keep_indices = mltest.run_op(ml,
                                 ml.device,
                                 True,
                                 ml.ops.nms,
                                 boxes,
                                 scores,
                                 nms_overlap_thresh=nms_overlap_thresh)

    np.testing.assert_equal(keep_indices, keep_indices_ref)


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
