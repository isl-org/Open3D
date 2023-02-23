// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/pytorch/misc/VoxelizeOpKernel.h"

#include "open3d/ml/impl/misc/Voxelize.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

using namespace open3d::ml::impl;

template <class T>
void VoxelizeCPU(const torch::Tensor& points,
                 const torch::Tensor& row_splits,
                 const torch::Tensor& voxel_size,
                 const torch::Tensor& points_range_min,
                 const torch::Tensor& points_range_max,
                 const int64_t max_points_per_voxel,
                 const int64_t max_voxels,
                 torch::Tensor& voxel_coords,
                 torch::Tensor& voxel_point_indices,
                 torch::Tensor& voxel_point_row_splits,
                 torch::Tensor& voxel_batch_splits) {
    VoxelizeOutputAllocator output_allocator(points.device().type(),
                                             points.device().index());

    switch (points.size(1)) {
#define CASE(NDIM)                                                            \
    case NDIM:                                                                \
        VoxelizeCPU<T, NDIM>(                                                 \
                points.size(0), points.data_ptr<T>(), row_splits.size(0) - 1, \
                row_splits.data_ptr<int64_t>(), voxel_size.data_ptr<T>(),     \
                points_range_min.data_ptr<T>(),                               \
                points_range_max.data_ptr<T>(), max_points_per_voxel,         \
                max_voxels, output_allocator);                                \
        break;
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        default:
            break;  // will be handled by the generic torch function

#undef CASE
    }

    voxel_coords = output_allocator.VoxelCoords();
    voxel_point_indices = output_allocator.VoxelPointIndices();
    voxel_point_row_splits = output_allocator.VoxelPointRowSplits();
    voxel_batch_splits = output_allocator.VoxelBatchSplits();
}

#define INSTANTIATE(T)                                                       \
    template void VoxelizeCPU<T>(                                            \
            const torch::Tensor& points, const torch::Tensor& row_splits,    \
            const torch::Tensor& voxel_size,                                 \
            const torch::Tensor& points_range_min,                           \
            const torch::Tensor& points_range_max,                           \
            const int64_t max_points_per_voxel, const int64_t max_voxels,    \
            torch::Tensor& voxel_coords, torch::Tensor& voxel_point_indices, \
            torch::Tensor& voxel_point_row_splits,                           \
            torch::Tensor& voxel_batch_splits);

INSTANTIATE(float)
INSTANTIATE(double)
