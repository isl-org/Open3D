// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL wrapper for the KnnSearch PyTorch op. Bridges torch::Tensor <->
// open3d::core::Tensor (zero-copy, via DLPack -- see TorchOpen3DBridge.h),
// installs PyTorch's current XPU queue as the ambient SYCL queue for the
// whole call, and delegates to o3dnns::KnnSearchSYCL.

#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/TorchOpen3DBridge.h"
#include "torch/script.h"

namespace o3dnns = open3d::core::nns;
using open3d::ml::torch_bridge::Open3DToTorchTensor;
using open3d::ml::torch_bridge::TorchToOpen3DTensor;

template <class T, class TIndex>
void KnnSearchSYCL(const torch::Tensor& points,
                   const torch::Tensor& queries,
                   const int64_t k,
                   const torch::Tensor& points_row_splits,
                   const torch::Tensor& queries_row_splits,
                   torch::Tensor& neighbors_index,
                   torch::Tensor& neighbors_row_splits,
                   torch::Tensor& neighbors_distance) {
    open3d::core::Tensor points_ = TorchToOpen3DTensor(points);
    open3d::core::Tensor queries_ = TorchToOpen3DTensor(queries);
    open3d::core::Tensor points_row_splits_ =
            TorchToOpen3DTensor(points_row_splits);
    open3d::core::Tensor queries_row_splits_ =
            TorchToOpen3DTensor(queries_row_splits);
    open3d::core::Tensor neighbors_index_, neighbors_distance_;
    // o3dnns::KnnSearchSYCL (like o3dnns::KnnSearchCUDA) requires
    // neighbors_row_splits to be pre-allocated by the caller (it only fills
    // it in), matching the pattern used by KnnIndex::SearchKnn.
    open3d::core::Tensor neighbors_row_splits_ = open3d::core::Tensor::Empty(
            {queries.size(0) + 1}, open3d::core::Int64);

    // Installs PyTorch's current XPU queue as the ambient queue for
    // points_'s device; o3dnns::KnnSearchSYCL (and everything it calls, via
    // core::sy::GetQueue) will run on this queue for the duration of this
    // scope.
    sycl::queue& torch_queue = c10::xpu::getCurrentXPUStream().queue();
    open3d::core::sy::SYCLScopedQueue scoped_queue(points_.GetDevice(),
                                                   torch_queue);

    o3dnns::KnnSearchSYCL<T, TIndex>(
            points_, points_row_splits_, queries_, queries_row_splits_, int(k),
            neighbors_index_, neighbors_row_splits_, neighbors_distance_,
            o3dnns::kSYCLKnnDefaultTileBytes);

    neighbors_index = Open3DToTorchTensor(neighbors_index_);
    neighbors_row_splits = Open3DToTorchTensor(neighbors_row_splits_);
    neighbors_distance = Open3DToTorchTensor(neighbors_distance_);
}

#define INSTANTIATE(T, TIndex)                                         \
    template void KnnSearchSYCL<T, TIndex>(                            \
            const torch::Tensor& points, const torch::Tensor& queries, \
            const int64_t k, const torch::Tensor& points_row_splits,   \
            const torch::Tensor& queries_row_splits,                   \
            torch::Tensor& neighbors_index,                            \
            torch::Tensor& neighbors_row_splits,                       \
            torch::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)
