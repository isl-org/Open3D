// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Reference:
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu
//
// Reference:
// https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/src/iou3d_kernel.cu
// 3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
// Written by Shaoshuai Shi
// All Rights Reserved 2019-2020.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "open3d/core/CUDAUtils.h"

#ifdef BUILD_SYCL_MODULE
// NmsSYCLKernel takes a real sycl::queue&, so any TU that sees this
// declaration needs the full SYCL runtime; NmsSYCL.cpp (built with -fsycl)
// and any *.cpp including this header are all compiled SYCL-aware when
// BUILD_SYCL_MODULE=ON (see pytorch/CMakeLists.txt / contrib/CMakeLists.txt).
#include <sycl/sycl.hpp>
#endif

#include "open3d/ml/contrib/IoUImpl.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace ml {
namespace contrib {

#ifdef BUILD_CUDA_MODULE

/// Runs NMS entirely on device and writes the kept box indices (in
/// score-descending order) into the caller-allocated \p keep_indices_out
/// (device memory, capacity >= n). The caller (NmsOps.cpp) passes in the
/// data_ptr() of a torch::empty({n}, ...) tensor already on the CUDA device,
/// so the (up to n-sized) result never leaves the device; only the returned
/// count -- a single int, not the index data -- is copied to the host, since
/// PyTorch tensor shapes must be known on the host.
/// \param boxes (n, 5) float32.
/// \param scores (n,) float32.
/// \param n Number of boxes.
/// \param nms_overlap_thresh When a high-score box is selected, other remaining
/// boxes with IoU > nms_overlap_thresh will be discarded.
/// \param keep_indices_out Output buffer, device memory, capacity >= n.
/// \return Number of boxes kept (i.e. valid entries written to
/// \p keep_indices_out).
int NmsCUDAKernel(const float *boxes,
                  const float *scores,
                  int n,
                  double nms_overlap_thresh,
                  int64_t *keep_indices_out);
#endif

#ifdef BUILD_SYCL_MODULE

/// SYCL counterpart of NmsCUDAKernel(); see its documentation. \p
/// keep_indices_out is device (USM) memory, typically the data_ptr() of a
/// torch::empty({n}, ...) tensor already on the SYCL/XPU device.
/// \param queue SYCL queue to run the kernel on.
/// \param boxes (n, 5) float32.
/// \param scores (n,) float32.
/// \param n Number of boxes.
/// \param nms_overlap_thresh When a high-score box is selected, other remaining
/// boxes with IoU > nms_overlap_thresh will be discarded.
/// \param keep_indices_out Output buffer, device (USM) memory, capacity >= n.
/// \return Number of boxes kept (i.e. valid entries written to
/// \p keep_indices_out).
int NmsSYCLKernel(sycl::queue &queue,
                  const float *boxes,
                  const float *scores,
                  int n,
                  double nms_overlap_thresh,
                  int64_t *keep_indices_out);
#endif

/// \param boxes (n, 5) float32.
/// \param scores (n,) float32.
/// \param n Number of boxes.
/// \param nms_overlap_thresh When a high-score box is selected, other remaining
/// boxes with IoU > nms_overlap_thresh will be discarded.
/// \return Selected box indices to keep.
std::vector<int64_t> NmsCPUKernel(const float *boxes,
                                  const float *scores,
                                  int n,
                                  double nms_overlap_thresh);

/// Sorts \p values in descending (or ascending) order and returns the
/// permutation indices. Shared by the CPU/CUDA/SYCL NMS launchers, which all
/// need a host-side ranking of the (small) scores array before running the
/// pairwise-overlap kernel.
template <typename T>
inline std::vector<int64_t> SortIndexesDescending(const T *values,
                                                   int64_t num) {
    std::vector<int64_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&values](int64_t i, int64_t j) {
                         return values[i] > values[j];
                     });
    return indices;
}

/// Greedily walks the sorted boxes and keeps a box iff it does not overlap
/// (per \p mask) with any previously-kept box, writing the kept *original*
/// box indices (in sorted/score-descending order) into the caller-allocated
/// \p keep_indices (capacity >= n). \p remv is caller-allocated scratch with
/// capacity >= num_block_cols = ceil(n / NMS_BLOCK_SIZE).
///
/// This is the single reduction shared, byte-for-byte, across the
/// CPU/CUDA/SYCL backends: the loop is inherently sequential (each step
/// depends on all previous ones via \p remv) and \p mask is tiny
/// (n x num_block_cols bits), so CUDA/SYCL run it as a single-thread device
/// kernel (see Nms.cu / NmsSYCL.cpp) rather than the host, keeping the whole
/// op on-device; CPU runs it directly (see NmsGreedyKeep below).
///
/// \param mask (n, num_block_cols) bitmask, mask[i, j] bit k is 1 iff
/// sorted box i overlaps with sorted box (NMS_BLOCK_SIZE*j + k).
/// \param sort_indices (n,) maps sorted position -> original box index.
/// \param n Number of boxes.
/// \param remv Scratch buffer, capacity >= ceil(n / NMS_BLOCK_SIZE).
/// \param keep_indices Output buffer, capacity >= n.
/// \return Number of boxes kept (i.e. the number of valid entries written
/// to \p keep_indices).
OPEN3D_HOST_DEVICE inline int NmsGreedyKeepCore(const uint64_t *mask,
                                                const int64_t *sort_indices,
                                                int n,
                                                uint64_t *remv,
                                                int64_t *keep_indices) {
    // Avoid utility::DivUp() (host-only, uses std::div) so this function
    // stays callable from CUDA/SYCL device code.
    const int num_block_cols = (n + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;
    for (int j = 0; j < num_block_cols; j++) {
        remv[j] = 0;
    }
    int count = 0;
    for (int i = 0; i < n; i++) {
        int block_col_idx = i / NMS_BLOCK_SIZE;
        int inner_block_col_idx = i % NMS_BLOCK_SIZE;

        if (!(remv[block_col_idx] & (1ULL << inner_block_col_idx))) {
            keep_indices[count++] = sort_indices[i];

            const uint64_t *p = mask + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv[j] |= p[j];
            }
        }
    }
    return count;
}

/// Host convenience wrapper around NmsGreedyKeepCore() for the CPU backend
/// (Nms.cpp), which already has \p mask and \p sort_indices in host memory.
inline std::vector<int64_t> NmsGreedyKeep(const uint64_t *mask,
                                          const int64_t *sort_indices,
                                          int n) {
    const int num_block_cols = utility::DivUp(n, NMS_BLOCK_SIZE);
    std::vector<uint64_t> remv(num_block_cols);
    std::vector<int64_t> keep_indices(n);
    int count = NmsGreedyKeepCore(mask, sort_indices, n, remv.data(),
                                  keep_indices.data());
    keep_indices.resize(count);
    return keep_indices;
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
