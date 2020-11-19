// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
//
#pragma once

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
void VoxelizeCPU(const torch::Tensor& points,
                 const torch::Tensor& voxel_size,
                 const torch::Tensor& points_range_min,
                 const torch::Tensor& points_range_max,
                 const int64_t max_points_per_voxel,
                 const int64_t max_voxels,
                 torch::Tensor& voxel_coords,
                 torch::Tensor& voxel_point_indices,
                 torch::Tensor& voxel_point_row_splits);

#ifdef BUILD_CUDA_MODULE
template <class T>
void VoxelizeCUDA(const torch::Tensor& points,
                  const torch::Tensor& voxel_size,
                  const torch::Tensor& points_range_min,
                  const torch::Tensor& points_range_max,
                  const int64_t max_points_per_voxel,
                  const int64_t max_voxels,
                  torch::Tensor& voxel_coords,
                  torch::Tensor& voxel_point_indices,
                  torch::Tensor& voxel_point_row_splits);
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

    const torch::Tensor& VoxelCoords() const { return voxel_coords; }
    const torch::Tensor& VoxelPointIndices() const {
        return voxel_point_indices;
    }
    const torch::Tensor& VoxelPointRowSplits() const {
        return voxel_point_row_splits;
    }

private:
    torch::Tensor voxel_coords;
    torch::Tensor voxel_point_indices;
    torch::Tensor voxel_point_row_splits;
    torch::DeviceType device_type;
    int device_idx;
};
