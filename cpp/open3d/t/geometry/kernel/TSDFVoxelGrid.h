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

#pragma once

#include <unordered_map>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {
void Touch(const core::Tensor& points,
           core::Tensor& voxel_block_coords,
           int64_t voxel_grid_resolution,
           float voxel_size,
           float sdf_trunc);

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
               float depth_max);

void ExtractSurfacePoints(const core::Tensor& block_indices,
                          const core::Tensor& nb_block_indices,
                          const core::Tensor& nb_block_masks,
                          const core::Tensor& block_keys,
                          const core::Tensor& block_values,
                          core::Tensor& points,
                          core::Tensor& normals,
                          core::Tensor& colors,
                          int64_t block_resolution,
                          float voxel_size);

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
                        float voxel_size);

void TouchCPU(const core::Tensor& points,
              core::Tensor& voxel_block_coords,
              int64_t voxel_grid_resolution,
              float voxel_size,
              float sdf_trunc);

void IntegrateCPU(const core::Tensor& depth,
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
                  float depth_max);

void ExtractSurfacePointsCPU(const core::Tensor& block_indices,
                             const core::Tensor& nb_block_indices,
                             const core::Tensor& nb_block_masks,
                             const core::Tensor& block_keys,
                             const core::Tensor& block_values,
                             core::Tensor& points,
                             core::Tensor& normals,
                             core::Tensor& colors,
                             int64_t block_resolution,
                             float voxel_size);

void ExtractSurfaceMeshCPU(const core::Tensor& block_indices,
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
                           float voxel_size);

#ifdef BUILD_CUDA_MODULE
void TouchCUDA(const core::Tensor& points,
               core::Tensor& voxel_block_coords,
               int64_t voxel_grid_resolution,
               float voxel_size,
               float sdf_trunc);

void IntegrateCUDA(const core::Tensor& depth,
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
                   float depth_max);

void ExtractSurfacePointsCUDA(const core::Tensor& block_indices,
                              const core::Tensor& nb_block_indices,
                              const core::Tensor& nb_block_masks,
                              const core::Tensor& block_keys,
                              const core::Tensor& block_values,
                              core::Tensor& points,
                              core::Tensor& normals,
                              core::Tensor& colors,
                              int64_t block_resolution,
                              float voxel_size);

void ExtractSurfaceMeshCUDA(const core::Tensor& block_indices,
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
                            float voxel_size);

#endif
}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
