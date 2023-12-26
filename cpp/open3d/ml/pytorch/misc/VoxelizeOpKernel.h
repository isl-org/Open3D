// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

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
                 torch::Tensor& voxel_batch_splits);

#ifdef BUILD_CUDA_MODULE
template <class T>
void VoxelizeCUDA(const torch::Tensor& points,
                  const torch::Tensor& row_splits,
                  const torch::Tensor& voxel_size,
                  const torch::Tensor& points_range_min,
                  const torch::Tensor& points_range_max,
                  const int64_t max_points_per_voxel,
                  const int64_t max_voxels,
                  torch::Tensor& voxel_coords,
                  torch::Tensor& voxel_point_indices,
                  torch::Tensor& voxel_point_row_splits,
                  torch::Tensor& voxel_batch_splits);
#endif

class VoxelizeOutputAllocator {
public:
    VoxelizeOutputAllocator(torch::DeviceType device_type, int device_idx)
        : device_type(device_type), device_idx(device_idx) {}

    void AllocVoxelCoords(int32_t** ptr, int64_t rows, int64_t cols) {
        voxel_coords = torch::empty({rows, cols},
                                    torch::dtype(ToTorchDtype<int32_t>())
                                            .device(device_type, device_idx));
        *ptr = voxel_coords.data_ptr<int32_t>();
    }

    void AllocVoxelPointIndices(int64_t** ptr, int64_t num) {
        voxel_point_indices =
                torch::empty({num}, torch::dtype(ToTorchDtype<int64_t>())
                                            .device(device_type, device_idx));
        *ptr = voxel_point_indices.data_ptr<int64_t>();
    }

    void AllocVoxelPointRowSplits(int64_t** ptr, int64_t num) {
        voxel_point_row_splits =
                torch::empty({num}, torch::dtype(ToTorchDtype<int64_t>())
                                            .device(device_type, device_idx));
        *ptr = voxel_point_row_splits.data_ptr<int64_t>();
    }

    void AllocVoxelBatchSplits(int64_t** ptr, int64_t num) {
        voxel_batch_splits =
                torch::empty({num}, torch::dtype(ToTorchDtype<int64_t>())
                                            .device(device_type, device_idx));
        *ptr = voxel_batch_splits.data_ptr<int64_t>();
    }

    const torch::Tensor& VoxelCoords() const { return voxel_coords; }
    const torch::Tensor& VoxelPointIndices() const {
        return voxel_point_indices;
    }
    const torch::Tensor& VoxelPointRowSplits() const {
        return voxel_point_row_splits;
    }
    const torch::Tensor& VoxelBatchSplits() const { return voxel_batch_splits; }

private:
    torch::Tensor voxel_coords;
    torch::Tensor voxel_point_indices;
    torch::Tensor voxel_point_row_splits;
    torch::Tensor voxel_batch_splits;
    torch::DeviceType device_type;
    int device_idx;
};
