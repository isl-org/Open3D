// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU implementation of RoiPool — mirrors RoiPoolKernel.cu's single-pass
// roipool3d_forward(..., boxes3d, ...) overload: for each (batch, box) pair,
// scan all points once, collect up to sampled_pts_num points inside the box
// (via the shared pt_in_box3d), and duplicate (modulo) collected points to
// pad up to sampled_pts_num. Parallelized over (batch, box) with TBB, since
// each pair is independent.

#include <tbb/parallel_for.h>

#include "open3d/ml/contrib/RoiPoolKernel.h"

namespace open3d {
namespace ml {
namespace contrib {

void roipool3dLauncherCPU(int batch_size,
                          int pts_num,
                          int boxes_num,
                          int feature_in_len,
                          int sampled_pts_num,
                          const float *xyz,
                          const float *boxes3d,
                          const float *pts_feature,
                          float *pooled_features,
                          int *pooled_empty_flag) {
    tbb::parallel_for(
            tbb::blocked_range<int>(0, batch_size * boxes_num),
            [&](const tbb::blocked_range<int> &r) {
                for (int idx = r.begin(); idx != r.end(); ++idx) {
                    const int i = idx / boxes_num;         // batch index
                    const int boxes_idx = idx % boxes_num;  // box index

                    int cnt = 0;
                    for (int k = 0; k < pts_num; k++) {
                        const int pt_offset = i * pts_num * 3 + k * 3;
                        const int box_offset =
                                i * boxes_num * 7 + boxes_idx * 7;

                        const int cur_in_flag = pt_in_box3d(
                                xyz[pt_offset], xyz[pt_offset + 1],
                                xyz[pt_offset + 2], boxes3d[box_offset],
                                boxes3d[box_offset + 1],
                                boxes3d[box_offset + 2],
                                boxes3d[box_offset + 3],
                                boxes3d[box_offset + 4],
                                boxes3d[box_offset + 5],
                                boxes3d[box_offset + 6], 10.0);
                        if (cur_in_flag) {
                            if (cnt < sampled_pts_num) {
                                const int feature_out_offset =
                                        i * boxes_num * sampled_pts_num *
                                                (3 + feature_in_len) +
                                        boxes_idx * sampled_pts_num *
                                                (3 + feature_in_len) +
                                        cnt * (3 + feature_in_len);
                                const int feature_in_offset =
                                        i * pts_num * feature_in_len +
                                        k * feature_in_len;

                                for (int j = 0; j < 3; j++)
                                    pooled_features[feature_out_offset + j] =
                                            xyz[pt_offset + j];
                                for (int j = 0; j < feature_in_len; j++)
                                    pooled_features[feature_out_offset + 3 +
                                                    j] =
                                            pts_feature[feature_in_offset +
                                                       j];

                                cnt++;
                            } else {
                                break;
                            }
                        }
                    }

                    if (cnt == 0) {
                        pooled_empty_flag[i * boxes_num + boxes_idx] = 1;
                    } else if (cnt < sampled_pts_num) {
                        // Duplicate same points for sampling.
                        for (int k = cnt; k < sampled_pts_num; k++) {
                            const int duplicate_idx = k % cnt;
                            const int src_offset =
                                    i * boxes_num * sampled_pts_num *
                                            (3 + feature_in_len) +
                                    boxes_idx * sampled_pts_num *
                                            (3 + feature_in_len) +
                                    duplicate_idx * (3 + feature_in_len);
                            const int dst_offset =
                                    i * boxes_num * sampled_pts_num *
                                            (3 + feature_in_len) +
                                    boxes_idx * sampled_pts_num *
                                            (3 + feature_in_len) +
                                    k * (3 + feature_in_len);
                            for (int j = 0; j < 3 + feature_in_len; j++)
                                pooled_features[dst_offset + j] =
                                        pooled_features[src_offset + j];
                        }
                    }
                }
            });
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
