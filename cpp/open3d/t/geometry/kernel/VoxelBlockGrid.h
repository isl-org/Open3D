// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <unordered_map>

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/TensorMap.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

using index_t = int;

void PointCloudTouch(std::shared_ptr<core::HashMap>& hashmap,
                     const core::Tensor& points,
                     core::Tensor& voxel_block_coords,
                     index_t voxel_grid_resolution,
                     float voxel_size,
                     float sdf_trunc);

void DepthTouch(std::shared_ptr<core::HashMap>& hashmap,
                const core::Tensor& depth,
                const core::Tensor& intrinsic,
                const core::Tensor& extrinsic,
                core::Tensor& voxel_block_coords,
                index_t voxel_grid_resolution,
                float voxel_size,
                float sdf_trunc,
                float depth_scale,
                float depth_max,
                index_t stride);

void GetVoxelCoordinatesAndFlattenedIndices(const core::Tensor& buf_indices,
                                            const core::Tensor& block_keys,
                                            core::Tensor& voxel_coords,
                                            core::Tensor& flattened_indices,
                                            index_t block_resolution,
                                            float voxel_size);

void Integrate(const core::Tensor& depth,
               const core::Tensor& color,
               const core::Tensor& block_indices,
               const core::Tensor& block_keys,
               TensorMap& block_value_map,
               const core::Tensor& depth_intrinsic,
               const core::Tensor& color_intrinsic,
               const core::Tensor& extrinsic,
               index_t resolution,
               float voxel_size,
               float sdf_trunc,
               float depth_scale,
               float depth_max);

void EstimateRange(const core::Tensor& block_keys,
                   core::Tensor& range_minmax_map,
                   const core::Tensor& intrinsics,
                   const core::Tensor& extrinsics,
                   int h,
                   int w,
                   int down_factor,
                   int64_t block_resolution,
                   float voxel_size,
                   float depth_min,
                   float depth_max,
                   core::Tensor& fragment_buffer);

void RayCast(std::shared_ptr<core::HashMap>& hashmap,
             const TensorMap& block_value_map,
             const core::Tensor& range_map,
             TensorMap& renderings_map,
             const core::Tensor& intrinsic,
             const core::Tensor& extrinsic,
             index_t h,
             index_t w,
             index_t block_resolution,
             float voxel_size,
             float depth_scale,
             float depth_min,
             float depth_max,
             float weight_threshold,
             float trunc_voxel_multiplier,
             int range_map_down_factor);

void ExtractPointCloud(const core::Tensor& block_indices,
                       const core::Tensor& nb_block_indices,
                       const core::Tensor& nb_block_masks,
                       const core::Tensor& block_keys,
                       const TensorMap& block_value_map,
                       core::Tensor& points,
                       core::Tensor& normals,
                       core::Tensor& colors,
                       index_t block_resolution,
                       float voxel_size,
                       float weight_threshold,
                       index_t& valid_size);

void ExtractTriangleMesh(const core::Tensor& block_indices,
                         const core::Tensor& inv_block_indices,
                         const core::Tensor& nb_block_indices,
                         const core::Tensor& nb_block_masks,
                         const core::Tensor& block_keys,
                         const TensorMap& block_value_map,
                         core::Tensor& vertices,
                         core::Tensor& triangles,
                         core::Tensor& vertex_normals,
                         core::Tensor& vertex_colors,
                         index_t block_resolution,
                         float voxel_size,
                         float weight_threshold,
                         index_t& vertex_count);

/// CPU
void PointCloudTouchCPU(std::shared_ptr<core::HashMap>& hashmap,
                        const core::Tensor& points,
                        core::Tensor& voxel_block_coords,
                        index_t voxel_grid_resolution,
                        float voxel_size,
                        float sdf_trunc);

void DepthTouchCPU(std::shared_ptr<core::HashMap>& hashmap,
                   const core::Tensor& depth,
                   const core::Tensor& intrinsic,
                   const core::Tensor& extrinsic,
                   core::Tensor& voxel_block_coords,
                   index_t voxel_grid_resolution,
                   float voxel_size,
                   float sdf_trunc,
                   float depth_scale,
                   float depth_max,
                   index_t stride);

void GetVoxelCoordinatesAndFlattenedIndicesCPU(const core::Tensor& buf_indices,
                                               const core::Tensor& block_keys,
                                               core::Tensor& voxel_coords,
                                               core::Tensor& flattened_indices,
                                               index_t block_resolution,
                                               float voxel_size);

template <typename input_depth_t,
          typename input_color_t,
          typename tsdf_t,
          typename weight_t,
          typename color_t>
void IntegrateCPU(const core::Tensor& depth,
                  const core::Tensor& color,
                  const core::Tensor& block_indices,
                  const core::Tensor& block_keys,
                  TensorMap& block_value_map,
                  const core::Tensor& depth_intrinsic,
                  const core::Tensor& color_intrinsic,
                  const core::Tensor& extrinsic,
                  index_t resolution,
                  float voxel_size,
                  float sdf_trunc,
                  float depth_scale,
                  float depth_max);

void EstimateRangeCPU(const core::Tensor& block_keys,
                      core::Tensor& range_minmax_map,
                      const core::Tensor& intrinsics,
                      const core::Tensor& extrinsics,
                      int h,
                      int w,
                      int down_factor,
                      int64_t block_resolution,
                      float voxel_size,
                      float depth_min,
                      float depth_max,
                      core::Tensor& fragment_buffer);

template <typename tsdf_t, typename weight_t, typename color_t>
void RayCastCPU(std::shared_ptr<core::HashMap>& hashmap,
                const TensorMap& block_value_map,
                const core::Tensor& range_map,
                TensorMap& renderings_map,
                const core::Tensor& intrinsic,
                const core::Tensor& extrinsic,
                index_t h,
                index_t w,
                index_t block_resolution,
                float voxel_size,
                float depth_scale,
                float depth_min,
                float depth_max,
                float weight_threshold,
                float trunc_voxel_multiplier,
                int range_map_down_factor);

template <typename tsdf_t, typename weight_t, typename color_t>
void ExtractPointCloudCPU(const core::Tensor& block_indices,
                          const core::Tensor& nb_block_indices,
                          const core::Tensor& nb_block_masks,
                          const core::Tensor& block_keys,
                          const TensorMap& block_value_map,
                          core::Tensor& points,
                          core::Tensor& normals,
                          core::Tensor& colors,
                          index_t block_resolution,
                          float voxel_size,
                          float weight_threshold,
                          index_t& valid_size);

template <typename tsdf_t, typename weight_t, typename color_t>
void ExtractTriangleMeshCPU(const core::Tensor& block_indices,
                            const core::Tensor& inv_block_indices,
                            const core::Tensor& nb_block_indices,
                            const core::Tensor& nb_block_masks,
                            const core::Tensor& block_keys,
                            const TensorMap& block_value_map,
                            core::Tensor& vertices,
                            core::Tensor& triangles,
                            core::Tensor& vertex_normals,
                            core::Tensor& vertex_colors,
                            index_t block_resolution,
                            float voxel_size,
                            float weight_threshold,
                            index_t& vertex_count);

#ifdef BUILD_CUDA_MODULE
void PointCloudTouchCUDA(std::shared_ptr<core::HashMap>& hashmap,
                         const core::Tensor& points,
                         core::Tensor& voxel_block_coords,
                         index_t voxel_grid_resolution,
                         float voxel_size,
                         float sdf_trunc);

void DepthTouchCUDA(std::shared_ptr<core::HashMap>& hashmap,
                    const core::Tensor& depth,
                    const core::Tensor& intrinsic,
                    const core::Tensor& extrinsic,
                    core::Tensor& voxel_block_coords,
                    index_t voxel_grid_resolution,
                    float voxel_size,
                    float sdf_trunc,
                    float depth_scale,
                    float depth_max,
                    index_t stride);

void GetVoxelCoordinatesAndFlattenedIndicesCUDA(const core::Tensor& buf_indices,
                                                const core::Tensor& block_keys,
                                                core::Tensor& voxel_coords,
                                                core::Tensor& flattened_indices,
                                                index_t block_resolution,
                                                float voxel_size);

template <typename input_depth_t,
          typename input_color_t,
          typename tsdf_t,
          typename weight_t,
          typename color_t>
void IntegrateCUDA(const core::Tensor& depth,
                   const core::Tensor& color,
                   const core::Tensor& block_indices,
                   const core::Tensor& block_keys,
                   TensorMap& block_value_map,
                   const core::Tensor& depth_intrinsic,
                   const core::Tensor& color_intrinsic,
                   const core::Tensor& extrinsic,
                   index_t resolution,
                   float voxel_size,
                   float sdf_trunc,
                   float depth_scale,
                   float depth_max);

void EstimateRangeCUDA(const core::Tensor& block_keys,
                       core::Tensor& range_minmax_map,
                       const core::Tensor& intrinsics,
                       const core::Tensor& extrinsics,
                       int h,
                       int w,
                       int down_factor,
                       int64_t block_resolution,
                       float voxel_size,
                       float depth_min,
                       float depth_max,
                       core::Tensor& fragment_buffer);

template <typename tsdf_t, typename weight_t, typename color_t>
void RayCastCUDA(std::shared_ptr<core::HashMap>& hashmap,
                 const TensorMap& block_value_map,
                 const core::Tensor& range_map,
                 TensorMap& renderings_map,
                 const core::Tensor& intrinsic,
                 const core::Tensor& extrinsic,
                 index_t h,
                 index_t w,
                 index_t block_resolution,
                 float voxel_size,
                 float depth_scale,
                 float depth_min,
                 float depth_max,
                 float weight_threshold,
                 float trunc_voxel_multiplier,
                 int range_map_down_factor);

template <typename tsdf_t, typename weight_t, typename color_t>
void ExtractPointCloudCUDA(const core::Tensor& block_indices,
                           const core::Tensor& nb_block_indices,
                           const core::Tensor& nb_block_masks,
                           const core::Tensor& block_keys,
                           const TensorMap& block_value_map,
                           core::Tensor& points,
                           core::Tensor& normals,
                           core::Tensor& colors,
                           index_t block_resolution,
                           float voxel_size,
                           float weight_threshold,
                           index_t& valid_size);

template <typename tsdf_t, typename weight_t, typename color_t>
void ExtractTriangleMeshCUDA(const core::Tensor& block_indices,
                             const core::Tensor& inv_block_indices,
                             const core::Tensor& nb_block_indices,
                             const core::Tensor& nb_block_masks,
                             const core::Tensor& block_keys,
                             const TensorMap& block_value_map,
                             core::Tensor& vertices,
                             core::Tensor& triangles,
                             core::Tensor& vertex_normals,
                             core::Tensor& vertex_colors,
                             index_t block_resolution,
                             float voxel_size,
                             float weight_threshold,
                             index_t& vertex_count);

#endif
}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
