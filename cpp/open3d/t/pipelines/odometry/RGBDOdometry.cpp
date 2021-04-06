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

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/kernel/RGBDOdometry.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

core::Tensor RGBDOdometryMultiScale(const t::geometry::RGBDImage& source,
                                    const t::geometry::RGBDImage& target,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    float depth_scale,
                                    float depth_max,
                                    float depth_diff,
                                    const std::vector<int>& iterations,
                                    const Method method) {
    // TODO (wei): more device check
    core::Device device = source.depth_.GetDevice();
    if (target.depth_.GetDevice() != device) {
        utility::LogError(
                "Device mismatch, got {} for source and {} for target.",
                device.ToString(), target.depth_.GetDevice().ToString());
    }

    core::Tensor intrinsics_d = intrinsics.To(device, true);

    // 4x4 transformations are always float64 and stay on CPU.
    core::Device host("CPU:0");
    core::Tensor trans_d = init_source_to_target.To(host, core::Dtype::Float64);

    core::Tensor source_depth_processed;
    core::Tensor target_depth_processed;
    kernel::odometry::PreprocessDepth(source.depth_.AsTensor(),
                                      source_depth_processed, depth_scale,
                                      depth_max);
    kernel::odometry::PreprocessDepth(target.depth_.AsTensor(),
                                      target_depth_processed, depth_scale,
                                      depth_max);

    int64_t n_levels = int64_t(iterations.size());
    if (method == Method::PointToPlane) {
        std::vector<core::Tensor> source_vertex_maps(n_levels);
        std::vector<core::Tensor> target_vertex_maps(n_levels);
        std::vector<core::Tensor> target_normal_maps(n_levels);
        std::vector<core::Tensor> intrinsic_matrices(n_levels);

        t::geometry::Image source_depth(source_depth_processed);
        t::geometry::Image target_depth(target_depth_processed);

        // Create image pyramid.
        for (int64_t i = 0; i < n_levels; ++i) {
            core::Tensor source_vertex_map =
                    CreateVertexMap(source_depth, intrinsics_d);

            core::Tensor target_vertex_map =
                    CreateVertexMap(target_depth, intrinsics_d);
            core::Tensor target_normal_map = CreateNormalMap(target_vertex_map);

            source_vertex_maps[n_levels - 1 - i] = source_vertex_map;
            target_vertex_maps[n_levels - 1 - i] = target_vertex_map;
            target_normal_maps[n_levels - 1 - i] = target_normal_map;

            intrinsic_matrices[n_levels - 1 - i] = intrinsics_d.Clone();

            if (i != n_levels - 1) {
                source_depth = t::geometry::Image(
                        PyrDownDepth(source_depth, depth_diff * 2));
                target_depth = t::geometry::Image(
                        PyrDownDepth(target_depth, depth_diff * 2));

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        for (int64_t i = 0; i < n_levels; ++i) {
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_source_to_target = ComputePosePointToPlane(
                        source_vertex_maps[i], target_vertex_maps[i],
                        target_normal_maps[i], intrinsic_matrices[i], trans_d,
                        depth_diff);
                trans_d = delta_source_to_target.Matmul(trans_d);
            }
        }
    } else if (method == Method::Hybrid) {
        std::vector<core::Tensor> source_intensity(n_levels);
        std::vector<core::Tensor> target_intensity(n_levels);

        std::vector<core::Tensor> source_depth(n_levels);
        std::vector<core::Tensor> target_depth(n_levels);
        std::vector<core::Tensor> target_intensity_dx(n_levels);
        std::vector<core::Tensor> target_intensity_dy(n_levels);

        std::vector<core::Tensor> target_depth_dx(n_levels);
        std::vector<core::Tensor> target_depth_dy(n_levels);

        std::vector<core::Tensor> source_vertex_maps(n_levels);

        std::vector<core::Tensor> intrinsic_matrices(n_levels);

        t::geometry::Image source_depth_curr(source_depth_processed);
        t::geometry::Image target_depth_curr(target_depth_processed);

        t::geometry::Image source_intensity_curr =
                source.color_.RGBToGray().To(core::Dtype::Float32);
        t::geometry::Image target_intensity_curr =
                target.color_.RGBToGray().To(core::Dtype::Float32);

        // Create image pyramid
        for (int64_t i = 0; i < n_levels; ++i) {
            source_depth[n_levels - 1 - i] =
                    source_depth_curr.AsTensor().Clone();
            target_depth[n_levels - 1 - i] =
                    target_depth_curr.AsTensor().Clone();

            source_intensity[n_levels - 1 - i] =
                    source_intensity_curr.AsTensor().Clone();
            target_intensity[n_levels - 1 - i] =
                    target_intensity_curr.AsTensor().Clone();

            core::Tensor source_vertex_map =
                    CreateVertexMap(source_depth_curr, intrinsics_d);
            source_vertex_maps[n_levels - 1 - i] = source_vertex_map;

            auto target_intensity_grad = target_intensity_curr.FilterSobel();
            target_intensity_dx[n_levels - 1 - i] =
                    target_intensity_grad.first.AsTensor();
            target_intensity_dy[n_levels - 1 - i] =
                    target_intensity_grad.second.AsTensor();

            auto target_depth_grad = target_depth_curr.FilterSobel();
            target_depth_dx[n_levels - 1 - i] =
                    target_depth_grad.first.AsTensor();
            target_depth_dy[n_levels - 1 - i] =
                    target_depth_grad.second.AsTensor();

            intrinsic_matrices[n_levels - 1 - i] = intrinsics_d.Clone();

            if (i != n_levels - 1) {
                source_depth_curr = source_depth_curr.PyrDown();
                target_depth_curr = target_depth_curr.PyrDown();
                source_intensity_curr = source_intensity_curr.PyrDown();
                target_intensity_curr = target_intensity_curr.PyrDown();

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        // Odometry
        for (int64_t i = 0; i < n_levels; ++i) {
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_source_to_target = ComputePoseHybrid(
                        source_depth[i], target_depth[i], source_intensity[i],
                        target_intensity[i], target_depth_dx[i],
                        target_depth_dy[i], target_intensity_dx[i],
                        target_intensity_dy[i], source_vertex_maps[i],
                        intrinsic_matrices[i], trans_d, depth_diff);
                trans_d = delta_source_to_target.Matmul(trans_d);
            }
        }
    } else if (method == Method::Intensity) {
        std::vector<core::Tensor> source_intensity(n_levels);
        std::vector<core::Tensor> target_intensity(n_levels);

        std::vector<core::Tensor> source_depth(n_levels);
        std::vector<core::Tensor> target_depth(n_levels);
        std::vector<core::Tensor> target_intensity_dx(n_levels);
        std::vector<core::Tensor> target_intensity_dy(n_levels);

        std::vector<core::Tensor> source_vertex_maps(n_levels);

        std::vector<core::Tensor> intrinsic_matrices(n_levels);

        t::geometry::Image source_depth_curr(source_depth_processed);
        t::geometry::Image target_depth_curr(target_depth_processed);

        t::geometry::Image source_intensity_curr =
                source.color_.RGBToGray().To(core::Dtype::Float32);
        t::geometry::Image target_intensity_curr =
                target.color_.RGBToGray().To(core::Dtype::Float32);

        // Create image pyramid
        for (int64_t i = 0; i < n_levels; ++i) {
            source_depth[n_levels - 1 - i] =
                    source_depth_curr.AsTensor().Clone();
            target_depth[n_levels - 1 - i] =
                    target_depth_curr.AsTensor().Clone();

            source_intensity[n_levels - 1 - i] =
                    source_intensity_curr.AsTensor().Clone();
            target_intensity[n_levels - 1 - i] =
                    target_intensity_curr.AsTensor().Clone();

            core::Tensor source_vertex_map =
                    CreateVertexMap(source_depth_curr, intrinsics_d);
            source_vertex_maps[n_levels - 1 - i] = source_vertex_map;

            auto target_intensity_grad = target_intensity_curr.FilterSobel();
            target_intensity_dx[n_levels - 1 - i] =
                    target_intensity_grad.first.AsTensor();
            target_intensity_dy[n_levels - 1 - i] =
                    target_intensity_grad.second.AsTensor();

            intrinsic_matrices[n_levels - 1 - i] = intrinsics_d.Clone();

            if (i != n_levels - 1) {
                source_depth_curr = t::geometry::Image(
                        PyrDownDepth(source_depth_curr, depth_diff * 2));
                target_depth_curr = t::geometry::Image(
                        PyrDownDepth(target_depth_curr, depth_diff * 2));
                source_intensity_curr = source_intensity_curr.PyrDown();
                target_intensity_curr = target_intensity_curr.PyrDown();

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        // Odometry
        for (int64_t i = 0; i < n_levels; ++i) {
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_source_to_target = ComputePoseIntensity(
                        source_depth[i], target_depth[i], source_intensity[i],
                        target_intensity[i], target_intensity_dx[i],
                        target_intensity_dy[i], source_vertex_maps[i],
                        intrinsic_matrices[i], trans_d, depth_diff);
                trans_d = delta_source_to_target.Matmul(trans_d);
            }
        }
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return trans_d;
}

core::Tensor ComputePosePointToPlane(const core::Tensor& source_vertex_map,
                                     const core::Tensor& target_vertex_map,
                                     const core::Tensor& target_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePosePointToPlane(
            source_vertex_map, target_vertex_map, target_normal_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);

    return pipelines::kernel::PoseToTransformation(se3_delta);
}

core::Tensor ComputePoseIntensity(const core::Tensor& source_depth,
                                  const core::Tensor& target_depth,
                                  const core::Tensor& source_intensity,
                                  const core::Tensor& target_intensity,
                                  const core::Tensor& target_intensity_dx,
                                  const core::Tensor& target_intensity_dy,
                                  const core::Tensor& source_vertex_map,
                                  const core::Tensor& intrinsics,
                                  const core::Tensor& init_source_to_target,
                                  float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePoseIntensity(
            source_depth, target_depth, source_intensity, target_intensity,
            target_intensity_dx, target_intensity_dy, source_vertex_map,
            intrinsics, init_source_to_target, se3_delta, residual, depth_diff);

    return pipelines::kernel::PoseToTransformation(se3_delta);
}

core::Tensor ComputePoseHybrid(const core::Tensor& source_depth,
                               const core::Tensor& target_depth,
                               const core::Tensor& source_intensity,
                               const core::Tensor& target_intensity,
                               const core::Tensor& target_depth_dx,
                               const core::Tensor& target_depth_dy,
                               const core::Tensor& target_intensity_dx,
                               const core::Tensor& target_intensity_dy,
                               const core::Tensor& source_vertex_map,
                               const core::Tensor& intrinsics,
                               const core::Tensor& init_source_to_target,
                               float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePoseHybrid(
            source_depth, target_depth, source_intensity, target_intensity,
            target_depth_dx, target_depth_dy, target_intensity_dx,
            target_intensity_dy, source_vertex_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);

    return pipelines::kernel::PoseToTransformation(se3_delta);
}

core::Tensor PreprocessDepth(const t::geometry::Image& depth,
                             float depth_scale,
                             float depth_max) {
    core::Tensor depth_processed;
    kernel::odometry::PreprocessDepth(depth.AsTensor(), depth_processed,
                                      depth_scale, depth_max);
    return depth_processed;
}

core::Tensor PyrDownDepth(const t::geometry::Image& depth, float depth_diff) {
    core::Tensor depth_down;
    kernel::odometry::PyrDownDepth(depth.AsTensor(), depth_down, depth_diff);
    return depth_down;
}

core::Tensor CreateVertexMap(const t::geometry::Image& depth,
                             const core::Tensor& intrinsics) {
    core::Tensor vertex_map;
    kernel::odometry::CreateVertexMap(depth.AsTensor(), intrinsics, vertex_map);
    return vertex_map;
}

core::Tensor CreateNormalMap(const core::Tensor& vertex_map) {
    core::Tensor normal_map;
    kernel::odometry::CreateNormalMap(vertex_map, normal_map);
    return normal_map;
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
