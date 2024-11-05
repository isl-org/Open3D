// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/nns/FixedRadiusSearchImpl.cuh"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::core::nns;

template <class T>
void BuildSpatialHashTableCUDA(const paddle::Tensor& points,
                               double radius,
                               const paddle::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               paddle::Tensor& hash_table_index,
                               paddle::Tensor& hash_table_cell_splits) {
    auto stream = points.stream();
    // -1 means current global place
    auto cuda_place_props = phi::backends::gpu::GetDeviceProperties(-1);
    const int texture_alignment = cuda_place_props.textureAlignment;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.shape()[0],
            points.data<T>(), T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.shape()[0],
            reinterpret_cast<uint32_t*>(const_cast<int32_t*>(
                    hash_table_cell_splits.data<int32_t>())),
            reinterpret_cast<uint32_t*>(
                    const_cast<int32_t*>(hash_table_index.data<int32_t>())));
    auto place = points.place();
    auto temp_tensor = CreateTempTensor(temp_size, place, &temp_ptr);

    // actually build the table
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.shape()[0],
            points.data<T>(), T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.shape()[0],
            reinterpret_cast<uint32_t*>(const_cast<int32_t*>(
                    hash_table_cell_splits.data<int32_t>())),
            reinterpret_cast<uint32_t*>(
                    const_cast<int32_t*>(hash_table_index.data<int32_t>())));
}

#define INSTANTIATE(T)                                            \
    template void BuildSpatialHashTableCUDA<T>(                   \
            const paddle::Tensor&, double, const paddle::Tensor&, \
            const std::vector<uint32_t>&, paddle::Tensor&, paddle::Tensor&);

INSTANTIATE(float)
