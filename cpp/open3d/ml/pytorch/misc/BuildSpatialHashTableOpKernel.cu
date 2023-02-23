// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "ATen/cuda/CUDAContext.h"
#include "open3d/core/nns/FixedRadiusSearchImpl.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

using namespace open3d::core::nns;

template <class T>
void BuildSpatialHashTableCUDA(const torch::Tensor& points,
                               double radius,
                               const torch::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               torch::Tensor& hash_table_index,
                               torch::Tensor& hash_table_cell_splits) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.size(0),
            points.data_ptr<T>(), T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());

    auto device = points.device();
    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually build the table
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.size(0),
            points.data_ptr<T>(), T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());
}

#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableCUDA<T>(                 \
            const torch::Tensor&, double, const torch::Tensor&, \
            const std::vector<uint32_t>&, torch::Tensor&, torch::Tensor&);

INSTANTIATE(float)
