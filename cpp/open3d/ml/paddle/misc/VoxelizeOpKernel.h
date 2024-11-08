// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <bits/stdint-intn.h>
#include <paddle/phi/common/place.h>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "paddle/extension.h"

template <class T>
void VoxelizeCPU(const paddle::Tensor& points,
                 const paddle::Tensor& row_splits,
                 const paddle::Tensor& voxel_size,
                 const paddle::Tensor& points_range_min,
                 const paddle::Tensor& points_range_max,
                 const int64_t max_points_per_voxel,
                 const int64_t max_voxels,
                 paddle::Tensor& voxel_coords,
                 paddle::Tensor& voxel_point_indices,
                 paddle::Tensor& voxel_point_row_splits,
                 paddle::Tensor& voxel_batch_splits);

#ifdef BUILD_CUDA_MODULE
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
                  paddle::Tensor& voxel_batch_splits);
#endif

class VoxelizeOutputAllocator {
public:
    VoxelizeOutputAllocator(paddle::Place place) : place(place) {}

    void AllocVoxelCoords(int32_t** ptr, int64_t rows, int64_t cols) {
        if (rows * cols == 0) {
            voxel_coords = InitializedEmptyTensor<int32_t>({rows, cols}, place);
        } else {
            voxel_coords = paddle::empty(
                    {rows, cols}, paddle::DataType(ToPaddleDtype<int32_t>()),
                    place);
        }
        *ptr = voxel_coords.data<int32_t>();
    }

    void AllocVoxelPointIndices(int64_t** ptr, int64_t num) {
        if (num == 0) {
            voxel_point_indices = InitializedEmptyTensor<int64_t>({num}, place);
        } else {
            voxel_point_indices = paddle::empty(
                    {num}, paddle::DataType(ToPaddleDtype<int64_t>()), place);
        }
        *ptr = voxel_point_indices.data<int64_t>();
    }

    void AllocVoxelPointRowSplits(int64_t** ptr, int64_t num) {
        voxel_point_row_splits = paddle::empty(
                {num}, paddle::DataType(ToPaddleDtype<int64_t>()), place);
        *ptr = voxel_point_row_splits.data<int64_t>();
    }

    void AllocVoxelBatchSplits(int64_t** ptr, int64_t num) {
        voxel_batch_splits = paddle::empty(
                {num}, paddle::DataType(ToPaddleDtype<int64_t>()), place);
        *ptr = voxel_batch_splits.data<int64_t>();
    }

    const paddle::Tensor& VoxelCoords() const { return voxel_coords; }
    const paddle::Tensor& VoxelPointIndices() const {
        return voxel_point_indices;
    }
    const paddle::Tensor& VoxelPointRowSplits() const {
        return voxel_point_row_splits;
    }
    const paddle::Tensor& VoxelBatchSplits() const {
        return voxel_batch_splits;
    }

private:
    paddle::Tensor voxel_coords;
    paddle::Tensor voxel_point_indices;
    paddle::Tensor voxel_point_row_splits;
    paddle::Tensor voxel_batch_splits;
    paddle::Place place;
};
