// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"

#include <vector>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

void PointCloudTouch(std::shared_ptr<core::HashMap>& hashmap,
                     const core::Tensor& points,
                     core::Tensor& voxel_block_coords,
                     int64_t voxel_grid_resolution,
                     float voxel_size,
                     float sdf_trunc) {
    core::Device device = points.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        PointCloudTouchCPU(hashmap, points, voxel_block_coords,
                           voxel_grid_resolution, voxel_size, sdf_trunc);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(PointCloudTouchCUDA, hashmap, points, voxel_block_coords,
                  voxel_grid_resolution, voxel_size, sdf_trunc);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void DepthTouch(std::shared_ptr<core::HashMap>& hashmap,
                const core::Tensor& depth,
                const core::Tensor& intrinsics,
                const core::Tensor& extrinsics,
                core::Tensor& voxel_block_coords,
                int64_t voxel_grid_resolution,
                float voxel_size,
                float sdf_trunc,
                float depth_scale,
                float depth_max,
                int stride) {
    core::Device device = depth.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        DepthTouchCPU(hashmap, depth, intrinsics, extrinsics,
                      voxel_block_coords, voxel_grid_resolution, voxel_size,
                      sdf_trunc, depth_scale, depth_max, stride);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(DepthTouchCUDA, hashmap, depth, intrinsics, extrinsics,
                  voxel_block_coords, voxel_grid_resolution, voxel_size,
                  sdf_trunc, depth_scale, depth_max, stride);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void Integrate(const core::Tensor& depth,
               const core::Tensor& color,
               const core::Tensor& block_indices,
               const core::Tensor& block_keys,
               std::vector<core::Tensor>& block_values,
               const core::Tensor& intrinsics,
               const core::Tensor& extrinsics,
               int64_t resolution,
               float voxel_size,
               float sdf_trunc,
               float depth_scale,
               float depth_max) {
    core::Device device = depth.GetDevice();

    core::Tensor depthf32 = depth.To(core::Float32);
    core::Tensor colorf32;

    if (color.NumElements() != 0) {
        if (color.GetDevice() != device) {
            utility::LogError(
                    "Incompatible color device type for depth and color");
        }
        colorf32 = color.To(core::Float32);
    }

    if (block_indices.GetDevice() != device ||
        block_keys.GetDevice() != device) {
        utility::LogError(
                "Incompatible device type for depth and TSDF voxel grid");
    }

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor extrinsics_d = extrinsics.To(host, core::Float64).Contiguous();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        IntegrateCPU(depthf32, colorf32, block_indices, block_keys,
                     block_values, intrinsics_d, extrinsics_d, resolution,
                     voxel_size, sdf_trunc, depth_scale, depth_max);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(IntegrateCUDA, depthf32, colorf32, block_indices, block_keys,
                  block_values, intrinsics_d, extrinsics_d, resolution,
                  voxel_size, sdf_trunc, depth_scale, depth_max);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void RayCast(std::shared_ptr<core::DeviceHashBackend>& hashmap,
             const std::vector<core::Tensor>& block_values,
             const core::Tensor& range_map,
             core::Tensor& vertex_map,
             core::Tensor& depth_map,
             core::Tensor& color_map,
             core::Tensor& normal_map,
             const core::Tensor& intrinsics,
             const core::Tensor& extrinsics,
             int h,
             int w,
             int64_t block_resolution,
             float voxel_size,
             float sdf_trunc,
             float depth_scale,
             float depth_min,
             float depth_max,
             float weight_threshold) {
    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor extrinsics_d = extrinsics.To(host, core::Float64).Contiguous();

    core::Device device = hashmap->GetDevice();
    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        RayCastCPU(hashmap, block_values, range_map, vertex_map, depth_map,
                   color_map, normal_map, intrinsics_d, extrinsics_d, h, w,
                   block_resolution, voxel_size, sdf_trunc, depth_scale,
                   depth_min, depth_max, weight_threshold);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(RayCastCUDA, hashmap, block_values, range_map, vertex_map,
                  depth_map, color_map, normal_map, intrinsics_d, extrinsics_d,
                  h, w, block_resolution, voxel_size, sdf_trunc, depth_scale,
                  depth_min, depth_max, weight_threshold);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void ExtractSurfacePoints(const core::Tensor& block_indices,
                          const core::Tensor& nb_block_indices,
                          const core::Tensor& nb_block_masks,
                          const core::Tensor& block_keys,
                          const std::vector<core::Tensor>& block_values,
                          core::Tensor& points,
                          core::Tensor& normals,
                          core::Tensor& colors,
                          int64_t block_resolution,
                          float voxel_size,
                          float weight_threshold,
                          int& valid_size) {
    core::Device device = block_keys.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ExtractSurfacePointsCPU(block_indices, nb_block_indices, nb_block_masks,
                                block_keys, block_values, points, normals,
                                colors, block_resolution, voxel_size,
                                weight_threshold, valid_size);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ExtractSurfacePointsCUDA, block_indices, nb_block_indices,
                  nb_block_masks, block_keys, block_values, points, normals,
                  colors, block_resolution, voxel_size, weight_threshold,
                  valid_size);
    } else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
