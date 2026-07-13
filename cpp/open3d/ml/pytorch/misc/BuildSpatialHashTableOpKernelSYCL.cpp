// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL wrapper for BuildSpatialHashTable PyTorch op. Delegates to
// BuildSpatialHashTableSYCLRaw from FixedRadiusSearchSYCLImpl.h, which
// implements the three-pass uniform-grid build (count → CSR scan → scatter)
// shared with the Open3D Tensor API in KnnSearchOpsSYCL.cpp.

#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "open3d/core/nns/kernel/FixedRadiusSearchSYCLImpl.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

/// Thin wrapper: extracts raw device pointers from PyTorch tensors, then
/// calls the shared BuildSpatialHashTableSYCLRaw kernel from the NNS layer.
template <class T>
void BuildSpatialHashTableSYCL(const torch::Tensor& points,
                               double radius,
                               const torch::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               torch::Tensor& hash_table_index,
                               torch::Tensor& hash_table_cell_splits) {
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    // points_row_splits is on CPU (moved by BuildSpatialHashTableOps.cpp).
    const int64_t* row_splits_ptr = points_row_splits.data_ptr<int64_t>();
    const int batch_size = static_cast<int>(points_row_splits.size(0)) - 1;

    open3d::core::nns::BuildSpatialHashTableSYCLRaw<T>(
            queue, points.data_ptr<T>(), T(1) / T(2 * radius), batch_size,
            row_splits_ptr, hash_table_splits.data(),
            reinterpret_cast<uint32_t*>(
                    hash_table_cell_splits.data_ptr<int32_t>()),
            static_cast<size_t>(hash_table_cell_splits.numel()),
            reinterpret_cast<uint32_t*>(hash_table_index.data_ptr<int32_t>()));
}

#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableSYCL<T>(                 \
            const torch::Tensor&, double, const torch::Tensor&, \
            const std::vector<uint32_t>&, torch::Tensor&, torch::Tensor&);

INSTANTIATE(float)
