// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"

#include <vector>

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {
void Touch(const core::Tensor& points,
           core::Tensor& voxel_block_coords,
           int64_t voxel_grid_resolution,
           float voxel_size,
           float sdf_trunc) {
    core::Device device = points.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        TouchCPU(points, voxel_block_coords, voxel_grid_resolution, voxel_size,
                 sdf_trunc);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        TouchCUDA(points, voxel_block_coords, voxel_grid_resolution, voxel_size,
                  sdf_trunc);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

void Integrate(const core::Tensor& depth,
               const core::Tensor& color,
               const core::Tensor& block_indices,
               const core::Tensor& block_keys,
               core::Tensor& block_values,
               const core::Tensor& intrinsics,
               const core::Tensor& extrinsics,
               int64_t resolution,
               float voxel_size,
               float sdf_trunc,
               float depth_scale,
               float depth_max) {
    core::Device device = depth.GetDevice();

    if (color.GetDevice() != device) {
        utility::LogError("Incompatible color device type for depth and color");
    }
    if (block_indices.GetDevice() != device ||
        block_keys.GetDevice() != device ||
        block_values.GetDevice() != device) {
        utility::LogError(
                "Incompatible device type for depth and TSDF voxel grid");
    }

    core::Tensor depthf32 = depth.To(core::Dtype::Float32);
    core::Tensor colorf32 = color.To(core::Dtype::Float32);
    core::Tensor intrinsicsf32 = intrinsics.To(device, core::Dtype::Float32);
    core::Tensor extrinsicsf32 = extrinsics.To(device, core::Dtype::Float32);

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        IntegrateCPU(depthf32, colorf32, block_indices, block_keys,
                     block_values, intrinsicsf32, extrinsicsf32, resolution,
                     voxel_size, sdf_trunc, depth_scale, depth_max);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        IntegrateCUDA(depthf32, colorf32, block_indices, block_keys,
                      block_values, intrinsicsf32, extrinsicsf32, resolution,
                      voxel_size, sdf_trunc, depth_scale, depth_max);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

void ExtractSurfacePoints(const core::Tensor& block_indices,
                          const core::Tensor& nb_block_indices,
                          const core::Tensor& nb_block_masks,
                          const core::Tensor& block_keys,
                          const core::Tensor& block_values,
                          core::Tensor& points,
                          core::Tensor& normals,
                          core::Tensor& colors,
                          int64_t block_resolution,
                          float voxel_size) {
    core::Device device = block_keys.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ExtractSurfacePointsCPU(block_indices, nb_block_indices, nb_block_masks,
                                block_keys, block_values, points, normals,
                                colors, block_resolution, voxel_size);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ExtractSurfacePointsCUDA(block_indices, nb_block_indices,
                                 nb_block_masks, block_keys, block_values,
                                 points, normals, colors, block_resolution,
                                 voxel_size);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

void ExtractSurfaceMesh(const core::Tensor& block_indices,
                        const core::Tensor& inv_block_indices,
                        const core::Tensor& nb_block_indices,
                        const core::Tensor& nb_block_masks,
                        const core::Tensor& block_keys,
                        const core::Tensor& block_values,
                        core::Tensor& vertices,
                        core::Tensor& triangles,
                        core::Tensor& vertex_normals,
                        core::Tensor& vertex_colors,
                        int64_t block_resolution,
                        float voxel_size) {
    core::Device device = block_keys.GetDevice();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ExtractSurfaceMeshCPU(block_indices, inv_block_indices,
                              nb_block_indices, nb_block_masks, block_keys,
                              block_values, vertices, triangles, vertex_normals,
                              vertex_colors, block_resolution, voxel_size);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ExtractSurfaceMeshCUDA(
                block_indices, inv_block_indices, nb_block_indices,
                nb_block_masks, block_keys, block_values, vertices, triangles,
                vertex_normals, vertex_colors, block_resolution, voxel_size);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}
}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
