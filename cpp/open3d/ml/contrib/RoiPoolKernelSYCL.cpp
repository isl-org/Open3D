// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of RoiPool — ports RoiPoolKernel.cu's 3-kernel
// pipeline (assign_pts_to_box3d -> get_pooled_idx -> roipool3d_forward),
// reusing the shared pt_in_box3d from RoiPoolKernel.h. pts_assign / pts_idx
// are USM device temporaries, mirroring the CUDA cudaMalloc scratch buffers.
//
// Kernel 2 (get_pooled_idx) uses one work-group per (batch, box); work-items
// grid-stride over pts_num in parallel and use an
// exclusive_scan_over_group-based compaction to claim output slots, instead
// of the CUDA/original-SYCL version's single-work-item serial scan. This
// changes which of the >sampled_pts_num assigned points are kept (the
// parallel scan does not visit points in strict ascending-index order the
// way a single serial scan does) -- see docs/dev/sycl_ml_ops_followups.md,
// intentionally deferred until the caller confirmed ordering doesn't matter
// (python/test/ml_ops/test_roi_pool.py now compares pooled feature sets per
// box, not slot-for-slot order).

#include "open3d/ml/contrib/RoiPoolKernel.h"

namespace open3d {
namespace ml {
namespace contrib {

namespace {
// Work-group size for kernel 2: one work-group per (batch, box), work-items
// grid-stride over pts_num. Best-guess default (matches the
// work-group-per-output-point size used elsewhere in this codebase, e.g.
// BallQuerySYCL.h/the conv FillColumn kernels); not yet tuned on target HW.
constexpr size_t kGetPooledIdxWGSize = 32;
}  // namespace

void roipool3dLauncherSYCL(sycl::queue &queue,
                           int batch_size,
                           int pts_num,
                           int boxes_num,
                           int feature_in_len,
                           int sampled_pts_num,
                           const float *xyz,
                           const float *boxes3d,
                           const float *pts_feature,
                           float *pooled_features,
                           int *pooled_empty_flag) {
    int *pts_assign = sycl::malloc_device<int>(
            static_cast<size_t>(batch_size) * pts_num * boxes_num, queue);

    // Kernel 1: for every (batch, point, box) triple, record whether the
    // point lies inside the box.
    sycl::event event1 = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(batch_size),
                               static_cast<size_t>(pts_num),
                               static_cast<size_t>(boxes_num)),
                [=](sycl::item<3> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int pt_idx = static_cast<int>(item.get_id(1));
                    const int box_idx = static_cast<int>(item.get_id(2));

                    const int assign_idx = bs_idx * pts_num * boxes_num +
                                           pt_idx * boxes_num + box_idx;
                    const int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
                    const int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

                    pts_assign[assign_idx] = pt_in_box3d(
                            xyz[pt_offset], xyz[pt_offset + 1],
                            xyz[pt_offset + 2], boxes3d[box_offset],
                            boxes3d[box_offset + 1], boxes3d[box_offset + 2],
                            boxes3d[box_offset + 3], boxes3d[box_offset + 4],
                            boxes3d[box_offset + 5], boxes3d[box_offset + 6],
                            10.0);
                });
    });

    int *pts_idx = sycl::malloc_device<int>(
            static_cast<size_t>(batch_size) * boxes_num * sampled_pts_num,
            queue);

    // Kernel 2: for every (batch, box), collect up to sampled_pts_num
    // assigned point indices via a work-group-cooperative parallel scan
    // (see file header), then pad (duplicate modulo cnt) if fewer than
    // sampled_pts_num points were assigned; flag boxes with zero points.
    const size_t wg2 = kGetPooledIdxWGSize;
    sycl::event event2 = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(event1);
        cgh.parallel_for(
                sycl::nd_range<1>(
                        sycl::range<1>(static_cast<size_t>(batch_size) *
                                       boxes_num * wg2),
                        sycl::range<1>(wg2)),
                [=](sycl::nd_item<1> item) {
                    const size_t group_id = item.get_group(0);
                    const int bs_idx = static_cast<int>(group_id / boxes_num);
                    const int boxes_idx =
                            static_cast<int>(group_id % boxes_num);
                    const size_t lid = item.get_local_id(0);
                    auto group = item.get_group();

                    const int *const assign_base =
                            pts_assign + bs_idx * pts_num * boxes_num +
                            boxes_idx;
                    int *const idx_out = pts_idx +
                                         bs_idx * boxes_num * sampled_pts_num +
                                         boxes_idx * sampled_pts_num;

                    int local_count = 0;
                    for (int k = static_cast<int>(lid); k < pts_num;
                         k += static_cast<int>(wg2)) {
                        if (assign_base[k * boxes_num]) ++local_count;
                    }

                    const int base_slot = sycl::exclusive_scan_over_group(
                            group, local_count, sycl::plus<int>());
                    const int total_count = sycl::reduce_over_group(
                            group, local_count, sycl::plus<int>());

                    if (lid == 0) {
                        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] =
                                (total_count == 0) ? 1 : 0;
                    }
                    if (total_count == 0) return;

                    int slot = base_slot;
                    for (int k = static_cast<int>(lid);
                         k < pts_num && slot < sampled_pts_num;
                         k += static_cast<int>(wg2)) {
                        if (assign_base[k * boxes_num]) {
                            idx_out[slot] = k;
                            ++slot;
                        }
                    }

                    // Padding (fewer than sampled_pts_num points assigned):
                    // duplicate already-collected indices modulo cnt,
                    // matching the CUDA/original-SYCL "duplicate_idx = k %
                    // cnt" pattern; cnt == total_count here. Sequential on
                    // one work-item since sampled_pts_num - total_count is
                    // typically small and each slot depends on an earlier
                    // one that may have been written by a different
                    // work-item (needs the barrier below first).
                    sycl::group_barrier(group);
                    if (lid == 0 && total_count < sampled_pts_num) {
                        for (int k = total_count; k < sampled_pts_num; ++k) {
                            const int duplicate_idx = k % total_count;
                            idx_out[k] = idx_out[duplicate_idx];
                        }
                    }
                });
    });

    // Kernel 3: gather xyz + features for each sampled point into the
    // output tensor; boxes with no assigned points are left as zeros.
    sycl::event event3 = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(event2);
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(batch_size),
                               static_cast<size_t>(boxes_num),
                               static_cast<size_t>(sampled_pts_num)),
                [=](sycl::item<3> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int box_idx = static_cast<int>(item.get_id(1));
                    const int sample_pt_idx = static_cast<int>(item.get_id(2));

                    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]) {
                        return;
                    }

                    const int temp_idx = bs_idx * boxes_num * sampled_pts_num +
                                         box_idx * sampled_pts_num +
                                         sample_pt_idx;
                    const int src_pt_idx = pts_idx[temp_idx];
                    const int dst_feature_offset =
                            temp_idx * (3 + feature_in_len);

                    for (int j = 0; j < 3; j++)
                        pooled_features[dst_feature_offset + j] =
                                xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

                    const int src_feature_offset =
                            bs_idx * pts_num * feature_in_len +
                            src_pt_idx * feature_in_len;
                    for (int j = 0; j < feature_in_len; j++)
                        pooled_features[dst_feature_offset + 3 + j] =
                                pts_feature[src_feature_offset + j];
                });
    });

    // pts_assign/pts_idx are USM scratch owned by this call; free them once
    // kernel 3 (the last reader of both) has completed. This is the one
    // blocking wait in the pipeline -- it exists to bound the scratch
    // buffers' lifetime, not as a lazy default between device-only stages.
    event3.wait();
    sycl::free(pts_assign, queue);
    sycl::free(pts_idx, queue);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
