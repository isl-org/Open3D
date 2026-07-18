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

#include "open3d/ml/contrib/RoiPoolKernel.h"

namespace open3d {
namespace ml {
namespace contrib {

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
    queue.submit([&](sycl::handler &cgh) {
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
                    const int box_offset =
                            bs_idx * boxes_num * 7 + box_idx * 7;
                    const int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

                    pts_assign[assign_idx] = pt_in_box3d(
                            xyz[pt_offset], xyz[pt_offset + 1],
                            xyz[pt_offset + 2], boxes3d[box_offset],
                            boxes3d[box_offset + 1], boxes3d[box_offset + 2],
                            boxes3d[box_offset + 3], boxes3d[box_offset + 4],
                            boxes3d[box_offset + 5], boxes3d[box_offset + 6],
                            10.0);
                });
    }).wait();

    int *pts_idx = sycl::malloc_device<int>(
            static_cast<size_t>(batch_size) * boxes_num * sampled_pts_num,
            queue);

    // Kernel 2: for every (batch, box), collect up to sampled_pts_num
    // assigned point indices, then pad (duplicate modulo cnt) if fewer than
    // sampled_pts_num points were assigned; flag boxes with zero points.
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
                sycl::range<2>(static_cast<size_t>(batch_size),
                              static_cast<size_t>(boxes_num)),
                [=](sycl::item<2> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int boxes_idx = static_cast<int>(item.get_id(1));

                    int cnt = 0;
                    for (int k = 0; k < pts_num; k++) {
                        if (pts_assign[bs_idx * pts_num * boxes_num +
                                      k * boxes_num + boxes_idx]) {
                            if (cnt < sampled_pts_num) {
                                pts_idx[bs_idx * boxes_num * sampled_pts_num +
                                       boxes_idx * sampled_pts_num + cnt] = k;
                                cnt++;
                            } else {
                                break;
                            }
                        }
                    }

                    if (cnt == 0) {
                        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
                    } else if (cnt < sampled_pts_num) {
                        const int base_offset =
                                bs_idx * boxes_num * sampled_pts_num +
                                boxes_idx * sampled_pts_num;
                        for (int k = cnt; k < sampled_pts_num; k++) {
                            const int duplicate_idx = k % cnt;
                            pts_idx[base_offset + k] =
                                    pts_idx[base_offset + duplicate_idx];
                        }
                    }
                });
    }).wait();

    // Kernel 3: gather xyz + features for each sampled point into the
    // output tensor; boxes with no assigned points are left as zeros.
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
                sycl::range<3>(static_cast<size_t>(batch_size),
                              static_cast<size_t>(boxes_num),
                              static_cast<size_t>(sampled_pts_num)),
                [=](sycl::item<3> item) {
                    const int bs_idx = static_cast<int>(item.get_id(0));
                    const int box_idx = static_cast<int>(item.get_id(1));
                    const int sample_pt_idx =
                            static_cast<int>(item.get_id(2));

                    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]) {
                        return;
                    }

                    const int temp_idx =
                            bs_idx * boxes_num * sampled_pts_num +
                            box_idx * sampled_pts_num + sample_pt_idx;
                    const int src_pt_idx = pts_idx[temp_idx];
                    const int dst_feature_offset =
                            temp_idx * (3 + feature_in_len);

                    for (int j = 0; j < 3; j++)
                        pooled_features[dst_feature_offset + j] =
                                xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 +
                                   j];

                    const int src_feature_offset =
                            bs_idx * pts_num * feature_in_len +
                            src_pt_idx * feature_in_len;
                    for (int j = 0; j < feature_in_len; j++)
                        pooled_features[dst_feature_offset + 3 + j] =
                                pts_feature[src_feature_offset + j];
                });
    });
    queue.wait_and_throw();

    sycl::free(pts_assign, queue);
    sycl::free(pts_idx, queue);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
