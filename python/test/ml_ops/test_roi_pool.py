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

    # roi_pool's SYCL implementation collects assigned points per box via a
    # parallel work-group scan rather than a serial index-ascending scan
    # (see RoiPoolKernelSYCL.cpp); when more than sampled_pts_num points are
    # assigned to a box, the subset kept (and the padding pattern when fewer
    # are assigned) can validly differ from a fixed reference file generated
    # by the old index-ascending algorithm, so verify correctness directly
    # against the input points/boxes instead: independently recompute
    # pt_in_box3d() membership in numpy, then check (a) pooled_empty_flag
    # matches whether any point is assigned to that box, and (b) for non-empty
    # boxes, every pooled xyz+feature row corresponds to some point actually
    # assigned to that box, and the number of *distinct* pooled points equals
    # min(sampled_pts_num, true assigned-point count).
    xyz = values0  # [batch, pts_num, 3]
    boxes3d = values1  # [batch, boxes_num, 7]: cx, bottom_y, cz, h, w, l, angle
    pts_feature = values2  # [batch, pts_num, feature_in_len]
    batch, _, _ = xyz.shape
    _, boxes_num, _ = boxes3d.shape
    max_dis = 10.0

    def pt_in_box3d(pts, box):
        cx, bottom_y, cz, h, w, l, angle = box
        cy = bottom_y - h / 2.0
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        outside = ((np.abs(x - cx) > max_dis) | (np.abs(y - cy) > h / 2.0) |
                   (np.abs(z - cz) > max_dis))
        cosa, sina = np.cos(angle), np.sin(angle)
        x_rot = (x - cx) * cosa + (z - cz) * (-sina)
        z_rot = (x - cx) * sina + (z - cz) * cosa
        inside = ((x_rot >= -l / 2.0) & (x_rot <= l / 2.0) &
                  (z_rot >= -w / 2.0) & (z_rot <= w / 2.0))
        return inside & ~outside

    for b in range(batch):
        for box in range(boxes_num):
            assigned_mask = pt_in_box3d(xyz[b], boxes3d[b, box])
            assigned_indices = np.nonzero(assigned_mask)[0]

            assert bool(ans1[b, box]) == (len(assigned_indices) == 0), (
                f"batch={b}, box={box}: pooled_empty_flag={ans1[b, box]} "
                f"but {len(assigned_indices)} points are assigned")
            if ans1[b, box]:
                continue  # empty box: pooled_features is all zeros either way

            pooled = ans0[b, box]  # [sampled_pts_num, 3 + feature_in_len]
            assigned_xyz = xyz[b, assigned_indices]
            assigned_feat = pts_feature[b, assigned_indices]
            assigned_rows = {
                tuple(np.concatenate([assigned_xyz[i], assigned_feat[i]]))
                for i in range(len(assigned_indices))
            }
            pooled_rows = [tuple(row) for row in pooled]
            for row in pooled_rows:
                assert row in assigned_rows, (
                    f"batch={b}, box={box}: pooled row {row} does not "
                    f"correspond to any point assigned to this box")
            distinct_count = len(set(pooled_rows))
            expected_count = min(sampled_pts_num, len(assigned_indices))
            assert distinct_count == expected_count, (
                f"batch={b}, box={box}: got {distinct_count} distinct "
                f"pooled points, expected {expected_count}")
