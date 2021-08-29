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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashMap.h"
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
    core::Device::DeviceType device_type = hashmap->GetDevice().GetType();

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
                const core::Tensor& intrinsic,
                const core::Tensor& extrinsic,
                core::Tensor& voxel_block_coords,
                int64_t voxel_grid_resolution,
                float voxel_size,
                float sdf_trunc,
                float depth_scale,
                float depth_max,
                int stride) {
    core::Device::DeviceType device_type = hashmap->GetDevice().GetType();

    if (device_type == core::Device::DeviceType::CPU) {
        DepthTouchCPU(hashmap, depth, intrinsic, extrinsic, voxel_block_coords,
                      voxel_grid_resolution, voxel_size, sdf_trunc, depth_scale,
                      depth_max, stride);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(DepthTouchCUDA, hashmap, depth, intrinsic, extrinsic,
                  voxel_block_coords, voxel_grid_resolution, voxel_size,
                  sdf_trunc, depth_scale, depth_max, stride);
    } else {
        utility::LogError("Unimplemented device");
    }
}

#define DISPATCH_VALUE_DTYPE_TO_TEMPLATE(WEIGHT_DTYPE, COLOR_DTYPE, ...)   \
    [&] {                                                                  \
        if (WEIGHT_DTYPE == open3d::core::Float32 &&                       \
            COLOR_DTYPE == open3d::core::Float32) {                        \
            using weight_t = float;                                        \
            using color_t = float;                                         \
            return __VA_ARGS__();                                          \
        } else if (WEIGHT_DTYPE == open3d::core::UInt16 &&                 \
                   COLOR_DTYPE == open3d::core::UInt16) {                  \
            using weight_t = uint16_t;                                     \
            using color_t = uint16_t;                                      \
            return __VA_ARGS__();                                          \
        } else {                                                           \
            utility::LogError("Unsupported value data type combination."); \
        }                                                                  \
    }()

#define DISPATCH_INPUT_DTYPE_TO_TEMPLATE(DEPTH_DTYPE, COLOR_DTYPE, ...)    \
    [&] {                                                                  \
        if (DEPTH_DTYPE == open3d::core::Float32 &&                        \
            COLOR_DTYPE == open3d::core::Float32) {                        \
            using input_depth_t = float;                                   \
            using input_color_t = float;                                   \
            return __VA_ARGS__();                                          \
        } else if (DEPTH_DTYPE == open3d::core::UInt16 &&                  \
                   COLOR_DTYPE == open3d::core::UInt8) {                   \
            using input_depth_t = uint16_t;                                \
            using input_color_t = uint8_t;                                 \
            return __VA_ARGS__();                                          \
        } else {                                                           \
            utility::LogError("Unsupported input data type combination."); \
        }                                                                  \
    }()

void Integrate(const core::Tensor& depth,
               const core::Tensor& color,
               const core::Tensor& block_indices,
               const core::Tensor& block_keys,
               std::vector<core::Tensor>& block_values,
               const core::Tensor& intrinsic,
               const core::Tensor& extrinsic,
               int64_t resolution,
               float voxel_size,
               float sdf_trunc,
               float depth_scale,
               float depth_max) {
    core::Device::DeviceType device_type = depth.GetDevice().GetType();

    using tsdf_t = float;
    DISPATCH_INPUT_DTYPE_TO_TEMPLATE(depth.GetDtype(), color.GetDtype(), [&] {
        DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
                block_values[1].GetDtype(), block_values[2].GetDtype(), [&] {
                    if (device_type == core::Device::DeviceType::CPU) {
                        IntegrateCPU<input_depth_t, input_color_t, tsdf_t,
                                     weight_t, color_t>(
                                depth, color, block_indices, block_keys,
                                block_values, intrinsic, extrinsic, resolution,
                                voxel_size, sdf_trunc, depth_scale, depth_max);
                    } else if (device_type == core::Device::DeviceType::CUDA) {
                        IntegrateCUDA<input_depth_t, input_color_t, tsdf_t,
                                      weight_t, color_t>(
                                depth, color, block_indices, block_keys,
                                block_values, intrinsic, extrinsic, resolution,
                                voxel_size, sdf_trunc, depth_scale, depth_max);
                    } else {
                        utility::LogError("Unimplemented device");
                    }
                });
    });
}

void RayCast(std::shared_ptr<core::HashMap>& hashmap,
             const std::vector<core::Tensor>& block_values,
             const core::Tensor& range_map,
             std::unordered_map<std::string, core::Tensor>& renderings_map,
             const core::Tensor& intrinsic,
             const core::Tensor& extrinsic,
             int h,
             int w,
             int64_t block_resolution,
             float voxel_size,
             float sdf_trunc,
             float depth_scale,
             float depth_min,
             float depth_max,
             float weight_threshold) {
    core::Device::DeviceType device_type = hashmap->GetDevice().GetType();

    using tsdf_t = float;
    DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
            block_values[1].GetDtype(), block_values[2].GetDtype(), [&] {
                if (device_type == core::Device::DeviceType::CPU) {
                    RayCastCPU<tsdf_t, weight_t, color_t>(
                            hashmap, block_values, range_map, renderings_map,
                            intrinsic, extrinsic, h, w, block_resolution,
                            voxel_size, sdf_trunc, depth_scale, depth_min,
                            depth_max, weight_threshold);
                } else if (device_type == core::Device::DeviceType::CUDA) {
                    RayCastCUDA<tsdf_t, weight_t, color_t>(
                            hashmap, block_values, range_map, renderings_map,
                            intrinsic, extrinsic, h, w, block_resolution,
                            voxel_size, sdf_trunc, depth_scale, depth_min,
                            depth_max, weight_threshold);
                } else {
                    utility::LogError("Unimplemented device");
                }
            });
}

void ExtractPointCloud(const core::Tensor& block_indices,
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
    core::Device::DeviceType device_type = block_indices.GetDevice().GetType();

    using tsdf_t = float;
    DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
            block_values[1].GetDtype(), block_values[2].GetDtype(), [&] {
                if (device_type == core::Device::DeviceType::CPU) {
                    ExtractPointCloudCPU<tsdf_t, weight_t, color_t>(
                            block_indices, nb_block_indices, nb_block_masks,
                            block_keys, block_values, points, normals, colors,
                            block_resolution, voxel_size, weight_threshold,
                            valid_size);
                } else if (device_type == core::Device::DeviceType::CUDA) {
                    ExtractPointCloudCUDA<tsdf_t, weight_t, color_t>(
                            block_indices, nb_block_indices, nb_block_masks,
                            block_keys, block_values, points, normals, colors,
                            block_resolution, voxel_size, weight_threshold,
                            valid_size);
                } else {
                    utility::LogError("Unimplemented device");
                }
            });
}

void ExtractTriangleMesh(const core::Tensor& block_indices,
                         const core::Tensor& inv_block_indices,
                         const core::Tensor& nb_block_indices,
                         const core::Tensor& nb_block_masks,
                         const core::Tensor& block_keys,
                         const std::vector<core::Tensor>& block_values,
                         core::Tensor& vertices,
                         core::Tensor& triangles,
                         core::Tensor& vertex_normals,
                         core::Tensor& vertex_colors,
                         int64_t block_resolution,
                         float voxel_size,
                         float weight_threshold,
                         int& vertex_count) {
    core::Device::DeviceType device_type = block_indices.GetDevice().GetType();

    using tsdf_t = float;
    DISPATCH_VALUE_DTYPE_TO_TEMPLATE(
            block_values[1].GetDtype(), block_values[2].GetDtype(), [&] {
                if (device_type == core::Device::DeviceType::CPU) {
                    ExtractTriangleMeshCPU<tsdf_t, weight_t, color_t>(
                            block_indices, inv_block_indices, nb_block_indices,
                            nb_block_masks, block_keys, block_values, vertices,
                            triangles, vertex_normals, vertex_colors,
                            block_resolution, voxel_size, weight_threshold,
                            vertex_count);
                } else if (device_type == core::Device::DeviceType::CUDA) {
                    ExtractTriangleMeshCUDA<tsdf_t, weight_t, color_t>(
                            block_indices, inv_block_indices, nb_block_indices,
                            nb_block_masks, block_keys, block_values, vertices,
                            triangles, vertex_normals, vertex_colors,
                            block_resolution, voxel_size, weight_threshold,
                            vertex_count);
                } else {
                    utility::LogError("Unimplemented device");
                }
            });
}

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
