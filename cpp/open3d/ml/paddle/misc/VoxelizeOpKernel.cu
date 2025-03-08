// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/impl/misc/Voxelize.cuh"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/VoxelizeOpKernel.h"
#include "paddle/extension.h"

using namespace open3d::ml::impl;

template <class T>
void VoxelizeCUDA(const paddle::Tensor& points,
                  const paddle::Tensor& row_splits,
                  const paddle::Tensor& voxel_size,
                  const paddle::Tensor& points_range_min,
                  const paddle::Tensor& points_range_max,
                  const int64_t max_points_per_voxel,
                  const int64_t max_voxels,
                  paddle::Tensor& voxel_coords,
                  paddle::Tensor& voxel_point_indices,
                  paddle::Tensor& voxel_point_row_splits,
                  paddle::Tensor& voxel_batch_splits) {
    auto stream = points.stream();
    // -1 means current global place
    auto cuda_device_props = phi::backends::gpu::GetDeviceProperties(-1);
    const int texture_alignment = cuda_device_props.textureAlignment;

    VoxelizeOutputAllocator output_allocator(points.place());

    switch (points.shape()[1]) {
#define CASE(NDIM)                                                            \
    case NDIM: {                                                              \
        void* temp_ptr = nullptr;                                             \
        size_t temp_size = 0;                                                 \
        VoxelizeCUDA<T, NDIM>(                                                \
                stream, temp_ptr, temp_size, texture_alignment,               \
                points.shape()[0], points.data<T>(),                          \
                row_splits.shape()[0] - 1, row_splits.data<int64_t>(),        \
                voxel_size.data<T>(), points_range_min.data<T>(),             \
                points_range_max.data<T>(), max_points_per_voxel, max_voxels, \
                output_allocator);                                            \
                                                                              \
        auto temp_tensor =                                                    \
                CreateTempTensor(temp_size, points.place(), &temp_ptr);       \
                                                                              \
        VoxelizeCUDA<T, NDIM>(                                                \
                stream, temp_ptr, temp_size, texture_alignment,               \
                points.shape()[0], points.data<T>(),                          \
                row_splits.shape()[0] - 1, row_splits.data<int64_t>(),        \
                voxel_size.data<T>(), points_range_min.data<T>(),             \
                points_range_max.data<T>(), max_points_per_voxel, max_voxels, \
                output_allocator);                                            \
    } break;
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        default:
            break;  // will be handled by the generic paddle function

#undef CASE
    }

    voxel_coords = output_allocator.VoxelCoords();
    voxel_point_indices = output_allocator.VoxelPointIndices();
    voxel_point_row_splits = output_allocator.VoxelPointRowSplits();
    voxel_batch_splits = output_allocator.VoxelBatchSplits();
}

#define INSTANTIATE(T)                                                         \
    template void VoxelizeCUDA<T>(                                             \
            const paddle::Tensor& points, const paddle::Tensor& row_splits,    \
            const paddle::Tensor& voxel_size,                                  \
            const paddle::Tensor& points_range_min,                            \
            const paddle::Tensor& points_range_max,                            \
            const int64_t max_points_per_voxel, const int64_t max_voxels,      \
            paddle::Tensor& voxel_coords, paddle::Tensor& voxel_point_indices, \
            paddle::Tensor& voxel_point_row_splits,                            \
            paddle::Tensor& voxel_batch_splits);

INSTANTIATE(float)
INSTANTIATE(double)
