// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of NMS — ports Nms.cu's NmsKernel and SortIndices.
// Mirrors the CUDA grid(num_block_cols, num_block_rows) x block(NMS_BLOCK_SIZE)
// layout as a single 1-D nd_range whose group index is unflattened into
// (block_row_idx, block_col_idx); a sycl::local_accessor holds the 64x5
// block of comparison boxes (matches the CUDA __shared__ block_boxes),
// including the same diagonal (dst_idx > src_idx) skip to avoid redundant
// comparisons. Ranking the (device-resident) scores uses oneDPL's
// std::stable_sort with a device execution policy, mirroring Nms.cu's
// thrust::stable_sort_by_key. The final greedy keep-loop (NmsGreedyKeepCore,
// shared with Nms.cu) is inherently sequential over a tiny
// (n x num_block_cols)-bit mask, so it is run as a single-work-item device
// kernel rather than unrolled on the host, writing straight into the
// caller-owned `keep_indices_out` (see NmsOps.cpp, which passes the
// data_ptr() of a device torch::Tensor). The whole op, including this
// reduction, stays on device; only the small `count` result (needed on the
// host to know the output tensor's shape) is copied back.
//
// `mask`/`remv`/`count` are kernel-private scratch (never touched by
// PyTorch or oneDPL), so they are backed by sycl::buffer rather than
// malloc_device/free -- the buffer destructor blocks until its last reader
// completes and releases the memory, giving the correct lifetime without a
// manual sycl::free() that could otherwise race a still-in-flight kernel.
// `sort_indices` must stay raw USM: it is sorted in place by
// oneapi::dpl::execution's device policy (which requires USM/shared
// pointers, not sycl::buffer), and is also read by the mask kernel and the
// final single_task.

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "open3d/ml/contrib/Nms.h"

namespace open3d {
namespace ml {
namespace contrib {

int NmsSYCLKernel(sycl::queue &queue,
                  const float *boxes,
                  const float *scores,
                  int n,
                  double nms_overlap_thresh,
                  int64_t *keep_indices_out) {
    if (n == 0) {
        return 0;
    }

    const int num_block_cols = utility::DivUp(n, NMS_BLOCK_SIZE);
    const int num_block_rows = num_block_cols;  // Grid is n x n in blocks.

    // Rank `scores` in descending order entirely on device: fill
    // sort_indices with 0..n-1, then stable_sort it by score via oneDPL
    // (mirrors Nms.cu's thrust::sequence + thrust::stable_sort_by_key).
    // stable_sort has no non-blocking oneDPL async equivalent, and the
    // device policy does not inherit a queue's pending work when the queue
    // is out-of-order (as PyTorch's XPU queue can be), so an explicit
    // barrier -- not a wait() -- makes the iota visible to it without
    // stalling the host.
    int64_t *sort_indices = sycl::malloc_device<int64_t>(n, queue);
    sycl::event iota_event = queue.parallel_for(
            sycl::range<1>(n),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                sort_indices[id[0]] = static_cast<int64_t>(id[0]);
            });
    queue.ext_oneapi_submit_barrier({iota_event});

    auto policy = oneapi::dpl::execution::make_device_policy(queue);
    std::stable_sort(
            policy, sort_indices, sort_indices + n,
            [scores](int64_t i, int64_t j) { return scores[i] > scores[j]; });

    // mask/remv/count are kernel-private scratch (never touched by PyTorch
    // or oneDPL), so they are backed by sycl::buffer rather than
    // malloc_device/free (see file header).
    sycl::buffer<uint64_t, 1> mask_buf{
            sycl::range<1>(static_cast<size_t>(n) * num_block_cols)};

    const size_t num_groups =
            static_cast<size_t>(num_block_rows) * num_block_cols;
    const sycl::range<1> global(num_groups * NMS_BLOCK_SIZE);
    const sycl::range<1> local(NMS_BLOCK_SIZE);

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> block_boxes(NMS_BLOCK_SIZE * 5, cgh);
        sycl::accessor mask_acc(mask_buf, cgh, sycl::write_only, sycl::no_init);

        cgh.parallel_for(
                sycl::nd_range<1>(global, local),
                // Distinct buffers — safe for [[intel::kernel_args_restrict]].
                [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
                    const int group_id = static_cast<int>(item.get_group(0));
                    const int block_row_idx = group_id / num_block_cols;
                    const int block_col_idx = group_id % num_block_cols;
                    const int lid = static_cast<int>(item.get_local_id(0));

                    const int row_size = sycl::min(
                            n - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
                    const int col_size = sycl::min(
                            n - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

                    // Stage the block_col_idx-th column block of boxes into
                    // local memory (matches CUDA's __shared__ block_boxes).
                    if (lid < col_size) {
                        float *dst = &block_boxes[lid * 5];
                        const int src_idx =
                                NMS_BLOCK_SIZE * block_col_idx + lid;
                        const float *src = boxes + sort_indices[src_idx] * 5;
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst[3] = src[3];
                        dst[4] = src[4];
                    }
                    sycl::group_barrier(item.get_group());

                    if (lid < row_size) {
                        const int src_idx =
                                NMS_BLOCK_SIZE * block_row_idx + lid;
                        // On the diagonal block, skip self- and
                        // already-compared pairs (dst_idx <= src local idx).
                        int dst_idx =
                                (block_row_idx == block_col_idx) ? lid + 1 : 0;

                        uint64_t t = 0;
                        while (dst_idx < col_size) {
                            if (IoUBev2DWithMinAndMax(
                                        boxes + sort_indices[src_idx] * 5,
                                        &block_boxes[dst_idx * 5]) >
                                nms_overlap_thresh) {
                                t |= 1ULL << dst_idx;
                            }
                            dst_idx++;
                        }
                        mask_acc[src_idx * num_block_cols + block_col_idx] = t;
                    }
                });
    });

    // Greedy keep-loop: run on-device as a single-work-item kernel (see file
    // header comment), writing straight into the caller-owned
    // `keep_indices_out` (no separate device allocation/free for the result,
    // and no device->host copy of the result data -- only the final count,
    // a single int, needs to reach the host).
    sycl::buffer<uint64_t, 1> remv_buf{sycl::range<1>(num_block_cols)};
    sycl::buffer<int, 1> count_buf{sycl::range<1>(1)};

    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor mask_acc(mask_buf, cgh, sycl::read_only);
        sycl::accessor remv_acc(remv_buf, cgh, sycl::write_only, sycl::no_init);
        sycl::accessor count_acc(count_buf, cgh, sycl::write_only,
                                 sycl::no_init);
        cgh.single_task([=]() {
            count_acc[0] = NmsGreedyKeepCore(&mask_acc[0], sort_indices, n,
                                             &remv_acc[0], keep_indices_out);
        });
    });

    // Reading `count` back to the host via a host_accessor is a genuine
    // synchronization point (PyTorch needs the tensor shape on the host);
    // it blocks until the single_task above (count_buf's only writer) has
    // completed, replacing the previous explicit memcpy + wait().
    sycl::host_accessor count_acc(count_buf, sycl::read_only);
    const int count = count_acc[0];

    // sort_indices is raw USM (see file header); the host_accessor read
    // above blocks on the single_task, whose sort_indices read happens
    // after the stable_sort above, so freeing it here is safe.
    sycl::free(sort_indices, queue);

    return count;
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
