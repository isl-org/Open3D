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

#pragma once

#include <unordered_map>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void Unproject(const core::Tensor& depth,
               utility::optional<std::reference_wrapper<const core::Tensor>>
                       image_colors,
               core::Tensor& points,
               utility::optional<std::reference_wrapper<core::Tensor>> colors,
               const core::Tensor& intrinsics,
               const core::Tensor& extrinsics,
               float depth_scale,
               float depth_max,
               int64_t stride);

void Project(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max);

void GetPointMaskWithinAABB(const core::Tensor& points,
                            const core::Tensor& min_bound,
                            const core::Tensor& max_bound,
                            core::Tensor& mask);

void GetPointMaskWithinOBB(const core::Tensor& points,
                           const core::Tensor& center,
                           const core::Tensor& rotation,
                           const core::Tensor& extent,
                           core::Tensor& mask);

void UnprojectCPU(
        const core::Tensor& depth,
        utility::optional<std::reference_wrapper<const core::Tensor>>
                image_colors,
        core::Tensor& points,
        utility::optional<std::reference_wrapper<core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        int64_t stride);

void ProjectCPU(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max);

void GetPointMaskWithinAABBCPU(const core::Tensor& points,
                               const core::Tensor& min_bound,
                               const core::Tensor& max_bound,
                               core::Tensor& mask);

void GetPointMaskWithinOBBCPU(const core::Tensor& points,
                              const core::Tensor& center,
                              const core::Tensor& rotation,
                              const core::Tensor& extent,
                              core::Tensor& mask);

void NormalizeNormalsCPU(core::Tensor& normals);

void OrientNormalsToAlignWithDirectionCPU(core::Tensor& normals,
                                          const core::Tensor& direction);

void OrientNormalsTowardsCameraLocationCPU(const core::Tensor& points,
                                           core::Tensor& normals,
                                           const core::Tensor& camera);

void ComputeBoundaryPointsCPU(const core::Tensor& points,
                              const core::Tensor& normals,
                              const core::Tensor& indices,
                              const core::Tensor& counts,
                              core::Tensor& mask,
                              double angle_threshold);

#ifdef BUILD_CUDA_MODULE
void UnprojectCUDA(
        const core::Tensor& depth,
        utility::optional<std::reference_wrapper<const core::Tensor>>
                image_colors,
        core::Tensor& points,
        utility::optional<std::reference_wrapper<core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max,
        int64_t stride);

void ProjectCUDA(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max);

void GetPointMaskWithinAABBCUDA(const core::Tensor& points,
                                const core::Tensor& min_bound,
                                const core::Tensor& max_bound,
                                core::Tensor& mask);

void GetPointMaskWithinOBBCUDA(const core::Tensor& points,
                               const core::Tensor& center,
                               const core::Tensor& rotation,
                               const core::Tensor& extent,
                               core::Tensor& mask);

void NormalizeNormalsCUDA(core::Tensor& normals);

void OrientNormalsToAlignWithDirectionCUDA(core::Tensor& normals,
                                           const core::Tensor& direction);

void OrientNormalsTowardsCameraLocationCUDA(const core::Tensor& points,
                                            core::Tensor& normals,
                                            const core::Tensor& camera);

void ComputeBoundaryPointsCUDA(const core::Tensor& points,
                               const core::Tensor& normals,
                               const core::Tensor& indices,
                               const core::Tensor& counts,
                               core::Tensor& mask,
                               double angle_threshold);
#endif

void EstimateCovariancesUsingHybridSearchCPU(const core::Tensor& points,
                                             core::Tensor& covariances,
                                             const double& radius,
                                             const int64_t& max_nn);

void EstimateCovariancesUsingKNNSearchCPU(const core::Tensor& points,
                                          core::Tensor& covariances,
                                          const int64_t& max_nn);

void EstimateCovariancesUsingRadiusSearchCPU(const core::Tensor& points,
                                             core::Tensor& covariances,
                                             const double& radius);

void EstimateNormalsFromCovariancesCPU(const core::Tensor& covariances,
                                       core::Tensor& normals,
                                       const bool has_normals);

void EstimateColorGradientsUsingHybridSearchCPU(const core::Tensor& points,
                                                const core::Tensor& normals,
                                                const core::Tensor& colors,
                                                core::Tensor& color_gradient,
                                                const double& radius,
                                                const int64_t& max_nn);

void EstimateColorGradientsUsingKNNSearchCPU(const core::Tensor& points,
                                             const core::Tensor& normals,
                                             const core::Tensor& colors,
                                             core::Tensor& color_gradient,
                                             const int64_t& max_nn);

void EstimateColorGradientsUsingRadiusSearchCPU(const core::Tensor& points,
                                                const core::Tensor& normals,
                                                const core::Tensor& colors,
                                                core::Tensor& color_gradient,
                                                const double& radius);

#ifdef BUILD_CUDA_MODULE
void EstimateCovariancesUsingHybridSearchCUDA(const core::Tensor& points,
                                              core::Tensor& covariances,
                                              const double& radius,
                                              const int64_t& max_nn);

void EstimateCovariancesUsingKNNSearchCUDA(const core::Tensor& points,
                                           core::Tensor& covariances,
                                           const int64_t& max_nn);

void EstimateCovariancesUsingRadiusSearchCUDA(const core::Tensor& points,
                                              core::Tensor& covariances,
                                              const double& radius);

void EstimateNormalsFromCovariancesCUDA(const core::Tensor& covariances,
                                        core::Tensor& normals,
                                        const bool has_normals);

void EstimateColorGradientsUsingHybridSearchCUDA(const core::Tensor& points,
                                                 const core::Tensor& normals,
                                                 const core::Tensor& colors,
                                                 core::Tensor& color_gradient,
                                                 const double& radius,
                                                 const int64_t& max_nn);

void EstimateColorGradientsUsingKNNSearchCUDA(const core::Tensor& points,
                                              const core::Tensor& normals,
                                              const core::Tensor& colors,
                                              core::Tensor& color_gradient,
                                              const int64_t& max_nn);

void EstimateColorGradientsUsingRadiusSearchCUDA(const core::Tensor& points,
                                                 const core::Tensor& normals,
                                                 const core::Tensor& colors,
                                                 core::Tensor& color_gradient,
                                                 const double& radius);
#endif

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
