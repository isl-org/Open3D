// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of RoiPool — ports RoiPoolKernel.cu's 3-kernel
// pipeline (assign_pts_to_box3d -> get_pooled_idx -> roipool3d_forward),
// reusing the shared pt_in_box3d from RoiPoolKernel.h. pts_assign / pts_idx
// are kernel-private scratch (never touched by PyTorch, oneDPL, or
// sycl-tla), backed by sycl::buffer so release waits for the last reader
// (USM malloc/free would need explicit sync across the three kernels).
//
// get_pooled_idx: one work-group per (batch, box); parallel grid-stride over
// pts_num with exclusive_scan_over_group slot assignment. When more than
// sampled_pts_num points are assigned, kept indices may differ in order from
// a serial scan; python/test/ml_ops/test_roi_pool.py compares per-box pooled
// feature sets, not slot-for-slot indices.

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
    // pts_assign is a per-(batch, point, box) inside/outside flag (0 or 1,
    // matching the CUDA int layout, not a packed bitmask -- pts_num is
    // typically in the thousands and boxes_num in the tens, so the O(1)
    // per-element cost dominates over the O(8x) memory-traffic saving a
    // bitmask would give; not worth the added indexing complexity here).
    sycl::buffer<int, 1> pts_assign_buf{sycl::range<1>(
            static_cast<size_t>(batch_size) * pts_num * boxes_num)};

    // Kernel 1: for every (batch, point, box) triple, record whether the
    // point lies inside the box.
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor pts_assign_acc(pts_assign_buf, cgh, sycl::write_only,
                                      sycl::no_init);
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(batch_size),
                               static_cast<size_t>(pts_num),
                               static_cast<size_t>(boxes_num)),
                // Distinct buffers — safe for [[intel::kernel_args_restrict]].
                [=](sycl::item<3> item) [[intel::kernel_args_restrict]] {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int pt_idx = static_cast<int>(item.get_id(1));
                    const int box_idx = static_cast<int>(item.get_id(2));

                    const int assign_idx = bs_idx * pts_num * boxes_num +
                                           pt_idx * boxes_num + box_idx;
                    const int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
                    const int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

                    pts_assign_acc[assign_idx] = pt_in_box3d(
                            xyz[pt_offset], xyz[pt_offset + 1],
                            xyz[pt_offset + 2], boxes3d[box_offset],
                            boxes3d[box_offset + 1], boxes3d[box_offset + 2],
                            boxes3d[box_offset + 3], boxes3d[box_offset + 4],
                            boxes3d[box_offset + 5], boxes3d[box_offset + 6],
                            10.0);
                });
    });

    sycl::buffer<int, 1> pts_idx_buf{sycl::range<1>(
            static_cast<size_t>(batch_size) * boxes_num * sampled_pts_num)};

    // Kernel 2: for every (batch, box), collect up to sampled_pts_num
    // assigned point indices via a work-group-cooperative parallel scan
    // (see file header), then pad (duplicate modulo cnt) if fewer than
    // sampled_pts_num points were assigned; flag boxes with zero points.
    // depends_on(event1) is not needed: the SYCL runtime tracks the
    // pts_assign_buf read-after-write dependency automatically from the
    // accessors below. pooled_empty_flag is raw USM (owned by the caller),
    // so kernel 3's read of it still needs an explicit event dependency.
    const size_t wg2 = kGetPooledIdxWGSize;
    sycl::event event2 = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor pts_assign_acc(pts_assign_buf, cgh, sycl::read_only);
        sycl::accessor pts_idx_acc(pts_idx_buf, cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(
                        sycl::range<1>(static_cast<size_t>(batch_size) *
                                       boxes_num * wg2),
                        sycl::range<1>(wg2)),
                // pooled_empty_flag is the only raw USM pointer here.
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
                    const size_t group_id = item.get_group(0);
                    const int bs_idx = static_cast<int>(group_id / boxes_num);
                    const int boxes_idx =
                            static_cast<int>(group_id % boxes_num);
                    const size_t lid = item.get_local_id(0);
                    auto group = item.get_group();

                    const size_t assign_base =
                            static_cast<size_t>(bs_idx) * pts_num * boxes_num +
                            boxes_idx;
                    const size_t idx_out_base =
                            static_cast<size_t>(bs_idx) * boxes_num *
                                    sampled_pts_num +
                            static_cast<size_t>(boxes_idx) * sampled_pts_num;

                    int local_count = 0;
                    for (int k = static_cast<int>(lid); k < pts_num;
                         k += static_cast<int>(wg2)) {
                        if (pts_assign_acc[assign_base +
                                           static_cast<size_t>(k) * boxes_num])
                            ++local_count;
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
                        if (pts_assign_acc[assign_base +
                                           static_cast<size_t>(k) *
                                                   boxes_num]) {
                            pts_idx_acc[idx_out_base + slot] = k;
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
                            pts_idx_acc[idx_out_base + k] =
                                    pts_idx_acc[idx_out_base + duplicate_idx];
                        }
                    }
                });
    });

    // Kernel 3 depends on kernel 2 explicitly: pooled_empty_flag is raw USM.
    queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(event2);
        sycl::accessor pts_idx_acc(pts_idx_buf, cgh, sycl::read_only);
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(batch_size),
                               static_cast<size_t>(boxes_num),
                               static_cast<size_t>(sampled_pts_num)),
                // Distinct buffers — safe for [[intel::kernel_args_restrict]].
                [=](sycl::item<3> item) [[intel::kernel_args_restrict]] {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int box_idx = static_cast<int>(item.get_id(1));
                    const int sample_pt_idx = static_cast<int>(item.get_id(2));

                    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]) {
                        return;
                    }

                    const int temp_idx = bs_idx * boxes_num * sampled_pts_num +
                                         box_idx * sampled_pts_num +
                                         sample_pt_idx;
                    const int src_pt_idx = pts_idx_acc[temp_idx];
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

    // pts_assign_buf/pts_idx_buf go out of scope here; their destructors
    // block until the last kernel reading them (event2 / kernel 3) has
    // completed, then release the memory -- this is the one blocking point
    // in the pipeline, matching the previous USM design's event3.wait(),
    // but without a separate sycl::free() that could race a still-in-flight
    // kernel.
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
