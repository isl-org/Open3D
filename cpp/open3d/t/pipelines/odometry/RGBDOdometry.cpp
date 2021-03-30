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
                                    float depth_diff,
                                    const std::vector<int>& iterations,
                                    const Method method) {
    // TODO: more device check
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

    // TODO: decouple interfaces
    if (method == Method::PointToPlane) {
        int64_t n = int64_t(iterations.size());

        std::vector<core::Tensor> source_vertex_maps(iterations.size());
        std::vector<core::Tensor> source_normal_maps(iterations.size());
        std::vector<core::Tensor> target_vertex_maps(iterations.size());
        std::vector<core::Tensor> intrinsic_matrices(iterations.size());

        t::geometry::Image source_depth = source.depth_;
        t::geometry::Image target_depth = target.depth_;

        // Create image pyramid.
        for (int64_t i = 0; i < n; ++i) {
            core::Tensor source_vertex_map =
                    CreateVertexMap(source_depth, intrinsics_d, depth_scale);
            core::Tensor source_normal_map = CreateNormalMap(source_vertex_map);

            core::Tensor target_vertex_map =
                    CreateVertexMap(target_depth, intrinsics_d, depth_scale);

            source_vertex_maps[n - 1 - i] = source_vertex_map;
            source_normal_maps[n - 1 - i] = source_normal_map;
            target_vertex_maps[n - 1 - i] = target_vertex_map;

            intrinsic_matrices[n - 1 - i] = intrinsics_d.Clone();

            if (i != n - 1) {
                source_depth = source_depth.PyrDown();
                target_depth = target_depth.PyrDown();

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        for (int64_t i = 0; i < n; ++i) {
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_source_to_target = ComputePosePointToPlane(
                        source_vertex_maps[i], target_vertex_maps[i],
                        source_normal_maps[i], intrinsic_matrices[i], trans_d,
                        depth_diff);
                trans_d = delta_source_to_target.Matmul(trans_d);
            }
        }
    } else if (method == Method::Hybrid) {
        int64_t n = int64_t(iterations.size());

        std::vector<core::Tensor> source_intensity(iterations.size());
        std::vector<core::Tensor> target_intensity(iterations.size());

        std::vector<core::Tensor> source_depth(iterations.size());
        std::vector<core::Tensor> target_depth(iterations.size());

        std::vector<core::Tensor> target_vertex_maps(iterations.size());
        std::vector<core::Tensor> source_intensity_dx(iterations.size());
        std::vector<core::Tensor> source_intensity_dy(iterations.size());

        std::vector<core::Tensor> source_depth_dx(iterations.size());
        std::vector<core::Tensor> source_depth_dy(iterations.size());

        std::vector<core::Tensor> intrinsic_matrices(iterations.size());

        core::Tensor source_depth_filtered;
        core::Tensor target_depth_filtered;
        kernel::odometry::PreprocessDepth(source.depth_.AsTensor(),
                                          source_depth_filtered, depth_scale,
                                          3.0);
        kernel::odometry::PreprocessDepth(target.depth_.AsTensor(),
                                          target_depth_filtered, depth_scale,
                                          3.0);
        t::geometry::Image source_depth_curr(source_depth_filtered);
        t::geometry::Image target_depth_curr(target_depth_filtered);

        t::geometry::Image source_intensity_curr =
                source.color_.RGBToGray().To(core::Dtype::Float32);
        t::geometry::Image target_intensity_curr =
                target.color_.RGBToGray().To(core::Dtype::Float32);

        // Create image pyramid
        for (int64_t i = 0; i < n; ++i) {
            source_depth[n - 1 - i] = source_depth_curr.AsTensor().Clone();
            target_depth[n - 1 - i] = target_depth_curr.AsTensor().Clone();

            source_intensity[n - 1 - i] =
                    source_intensity_curr.AsTensor().Clone();
            target_intensity[n - 1 - i] =
                    target_intensity_curr.AsTensor().Clone();

            core::Tensor target_vertex_map = CreateVertexMap(
                    target_depth_curr, intrinsics_d, depth_scale);
            target_vertex_maps[n - 1 - i] = target_vertex_map;

            auto source_intensity_grad = source_intensity_curr.FilterSobel();
            source_intensity_dx[n - 1 - i] =
                    source_intensity_grad.first.AsTensor();
            source_intensity_dy[n - 1 - i] =
                    source_intensity_grad.second.AsTensor();

            auto source_depth_grad = source_depth_curr.FilterSobel();
            source_depth_dx[n - 1 - i] = source_depth_grad.first.AsTensor();
            source_depth_dy[n - 1 - i] = source_depth_grad.second.AsTensor();

            intrinsic_matrices[n - 1 - i] = intrinsics_d.Clone();

            if (i != n - 1) {
                source_depth_curr = source_depth_curr.PyrDown();
                target_depth_curr = target_depth_curr.PyrDown();
                source_intensity_curr = source_intensity_curr.PyrDown();
                target_intensity_curr = target_intensity_curr.PyrDown();

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        // Odometry
        for (int64_t i = 0; i < n; ++i) {
            for (int iter = 0; iter < iterations[i]; ++iter) {
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_intensity_dx[i])
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_intensity_dy[i])
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_depth_dx[i] /
                //                 1000.0)
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_depth_dy[i] /
                //                 1000.0)
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_depth[i] / 5000.0)
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(source_intensity[i])
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(target_depth[i] / 5000.0)
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(target_intensity[i])
                //                         .ToLegacyImage())});
                // visualization::DrawGeometries(
                //         {std::make_shared<open3d::geometry::Image>(
                //                 t::geometry::Image(target_vertex_maps[i])
                //                         .ToLegacyImage())});

                core::Tensor delta_source_to_target = ComputePoseHybrid(
                        source_depth[i], target_depth[i], source_intensity[i],
                        target_intensity[i], source_depth_dx[i],
                        source_depth_dy[i], source_intensity_dx[i],
                        source_intensity_dy[i], target_vertex_maps[i],
                        intrinsic_matrices[i], trans_d, depth_diff);
                trans_d = delta_source_to_target.Matmul(trans_d);
            }
        }
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return trans_d;
}

core::Tensor CreateVertexMap(const t::geometry::Image& depth,
                             const core::Tensor& intrinsics,
                             float depth_scale,
                             float depth_max) {
    core::Tensor vertex_map;
    kernel::odometry::CreateVertexMap(depth.AsTensor(), intrinsics, vertex_map,
                                      depth_scale, depth_max);
    return vertex_map;
}

core::Tensor CreateNormalMap(const core::Tensor& vertex_map) {
    core::Tensor normal_map;
    kernel::odometry::CreateNormalMap(vertex_map, normal_map);
    return normal_map;
}

core::Tensor ComputePosePointToPlane(const core::Tensor& source_vertex_map,
                                     const core::Tensor& target_vertex_map,
                                     const core::Tensor& source_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePosePointToPlane(
            source_vertex_map, target_vertex_map, source_normal_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);

    core::Tensor T_delta_inv =
            pipelines::kernel::PoseToTransformation(se3_delta);

    // T.inv = [R.T | -R.T @ t]
    core::Tensor R_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 0, 3);
    core::Tensor t_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 3, 4);

    core::Tensor T_delta = core::Tensor::Zeros({4, 4}, core::Dtype::Float64);
    T_delta.Slice(0, 0, 3).Slice(1, 0, 3) = R_inv.T();
    T_delta.Slice(0, 0, 3).Slice(1, 3, 4) = R_inv.T().Matmul(t_inv).Neg();
    T_delta[-1][-1] = 1;

    return T_delta;
}

core::Tensor ComputePoseHybrid(const core::Tensor& source_depth,
                               const core::Tensor& target_depth,
                               const core::Tensor& source_intensity,
                               const core::Tensor& target_intensity,
                               const core::Tensor& source_depth_dx,
                               const core::Tensor& source_depth_dy,
                               const core::Tensor& source_intensity_dx,
                               const core::Tensor& source_intensity_dy,
                               const core::Tensor& target_vertex_map,
                               const core::Tensor& intrinsics,
                               const core::Tensor& init_source_to_target,
                               float depth_diff) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePoseHybrid(
            source_depth, target_depth, source_intensity, target_intensity,
            source_depth_dx, source_depth_dy, source_intensity_dx,
            source_intensity_dy, target_vertex_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);

    core::Tensor T_delta_inv =
            pipelines::kernel::PoseToTransformation(se3_delta);

    // T.inv = [R.T | -R.T @ t]
    core::Tensor R_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 0, 3);
    core::Tensor t_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 3, 4);

    core::Tensor T_delta = core::Tensor::Zeros({4, 4}, core::Dtype::Float64);
    T_delta.Slice(0, 0, 3).Slice(1, 0, 3) = R_inv.T();
    T_delta.Slice(0, 0, 3).Slice(1, 3, 4) = R_inv.T().Matmul(t_inv).Neg();
    T_delta[-1][-1] = 1;

    return T_delta;
}

core::Tensor ComputePoseIntensity(const core::Tensor& source_vertex_map,
                                  const core::Tensor& target_vertex_map,
                                  const core::Tensor& source_color,
                                  const core::Tensor& target_color,
                                  const core::Tensor& source_color_dx,
                                  const core::Tensor& source_color_dy,
                                  const core::Tensor& intrinsics,
                                  const core::Tensor& init_source_to_target,
                                  float depth_diff) {
    utility::LogError("Direct intensity odometry unimplemented.");
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
