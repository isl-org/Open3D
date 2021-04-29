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
#include "open3d/t/geometry/kernel/Image.h"
#include "open3d/t/pipelines/kernel/RGBDOdometry.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

namespace {
t::geometry::Image PyrDownDepth(t::geometry::Image& src,
                                float diff_threshold,
                                float invalid_fill) {
    if (src.GetRows() <= 0 || src.GetCols() <= 0 || src.GetChannels() != 1) {
        utility::LogError(
                "Invalid shape, expected a 1 channel image, but got ({}, {}, "
                "{})",
                src.GetRows(), src.GetCols(), src.GetChannels());
    }
    if (src.GetDtype() != core::Dtype::Float32) {
        utility::LogError("Expected a Float32 image, but got {}",
                          src.GetDtype().ToString());
    }

    core::Tensor dst_tensor =
            core::Tensor::Empty({src.GetRows() / 2, src.GetCols() / 2, 1},
                                src.GetDtype(), src.GetDevice());
    t::geometry::kernel::image::PyrDownDepth(src.AsTensor(), dst_tensor,
                                             diff_threshold, invalid_fill);
    return t::geometry::Image(dst_tensor);
}
}  // namespace

OdometryResult RGBDOdometryMultiScalePointToPlane(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsics,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria);

OdometryResult RGBDOdometryMultiScaleIntensity(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsic,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria);

OdometryResult RGBDOdometryMultiScaleHybrid(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsics,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria);

OdometryResult RGBDOdometryMultiScale(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const Method method,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria) {
    // TODO (wei): more device check
    core::Device device = source.depth_.GetDevice();
    if (target.depth_.GetDevice() != device) {
        utility::LogError(
                "Device mismatch, got {} for source and {} for target.",
                device.ToString(), target.depth_.GetDevice().ToString());
    }

    // 4x4 transformations are always float64 and stay on CPU.
    core::Device host("CPU:0");
    core::Tensor intrinsics_d =
            intrinsics.To(host, core::Dtype::Float64).Clone();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Dtype::Float64).Clone();

    t::geometry::Image source_depth = source.depth_;
    t::geometry::Image target_depth = target.depth_;

    t::geometry::Image source_depth_processed =
            source_depth.ClipTransform(depth_scale, 0, depth_max, NAN);
    t::geometry::Image target_depth_processed =
            target_depth.ClipTransform(depth_scale, 0, depth_max, NAN);

    t::geometry::RGBDImage source_processed(source.color_,
                                            source_depth_processed);
    t::geometry::RGBDImage target_processed(target.color_,
                                            target_depth_processed);

    if (method == Method::PointToPlane) {
        return RGBDOdometryMultiScalePointToPlane(
                source_processed, target_processed, intrinsics_d, trans_d,
                depth_scale, depth_max, iterations, params, criteria);
    } else if (method == Method::Intensity) {
        return RGBDOdometryMultiScaleIntensity(
                source_processed, target_processed, intrinsics_d, trans_d,
                depth_scale, depth_max, iterations, params, criteria);
    } else if (method == Method::Hybrid) {
        return RGBDOdometryMultiScaleHybrid(
                source_processed, target_processed, intrinsics_d, trans_d,
                depth_scale, depth_max, iterations, params, criteria);
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return OdometryResult(trans_d);
}

OdometryResult RGBDOdometryMultiScalePointToPlane(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsics,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria) {
    int64_t n_levels = int64_t(iterations.size());
    std::vector<core::Tensor> source_vertex_maps(n_levels);
    std::vector<core::Tensor> target_vertex_maps(n_levels);
    std::vector<core::Tensor> target_normal_maps(n_levels);
    std::vector<core::Tensor> intrinsic_matrices(n_levels);

    t::geometry::Image source_depth_curr = source.depth_;
    t::geometry::Image target_depth_curr = target.depth_;

    // Create image pyramid.
    for (int64_t i = 0; i < n_levels; ++i) {
        t::geometry::Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics, NAN);
        t::geometry::Image target_vertex_map =
                target_depth_curr.CreateVertexMap(intrinsics, NAN);

        t::geometry::Image target_depth_curr_smooth =
                target_depth_curr.FilterBilateral(5, 5, 10);
        t::geometry::Image target_vertex_map_smooth =
                target_depth_curr_smooth.CreateVertexMap(intrinsics, NAN);
        t::geometry::Image target_normal_map =
                target_vertex_map_smooth.CreateNormalMap(NAN);

        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();
        target_vertex_maps[n_levels - 1 - i] = target_vertex_map.AsTensor();
        target_normal_maps[n_levels - 1 - i] = target_normal_map.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics.Clone();

        if (i != n_levels - 1) {
            source_depth_curr = PyrDownDepth(
                    source_depth_curr, params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = PyrDownDepth(
                    target_depth_curr, params.depth_outlier_trunc_ * 2, NAN);

            intrinsics /= 2;
            intrinsics[-1][-1] = 1;
        }
    }

    OdometryResult result;
    double prev_rmse = 0.0;
    double prev_fitness = 1.0;
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < iterations[i]; ++iter) {
            result = ComputePosePointToPlane(
                    source_vertex_maps[i], target_vertex_maps[i],
                    target_normal_maps[i], intrinsic_matrices[i], trans,
                    params.depth_outlier_trunc_, params.depth_huber_delta_);

            core::Tensor delta_source_to_target = result.transformation_;
            trans = delta_source_to_target.Matmul(trans).Contiguous();
            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, result.inlier_rmse_, result.fitness_);

            if (std::abs(prev_fitness - result.fitness_) / prev_fitness <
                        criteria.relative_fitness_ &&
                std::abs(prev_rmse - result.inlier_rmse_) / prev_rmse <
                        criteria.relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            prev_rmse = result.inlier_rmse_;
            prev_fitness = result.fitness_;
        }
    }

    return OdometryResult(trans, result.inlier_rmse_, result.fitness_);
}

OdometryResult RGBDOdometryMultiScaleIntensity(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsics,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria) {
    int64_t n_levels = int64_t(iterations.size());
    std::vector<core::Tensor> source_intensity(n_levels);
    std::vector<core::Tensor> target_intensity(n_levels);

    std::vector<core::Tensor> source_depth(n_levels);
    std::vector<core::Tensor> target_depth(n_levels);
    std::vector<core::Tensor> target_intensity_dx(n_levels);
    std::vector<core::Tensor> target_intensity_dy(n_levels);

    std::vector<core::Tensor> source_vertex_maps(n_levels);

    std::vector<core::Tensor> intrinsic_matrices(n_levels);

    t::geometry::Image source_depth_curr = source.depth_;
    t::geometry::Image target_depth_curr = target.depth_;

    t::geometry::Image source_intensity_curr =
            source.color_.RGBToGray().To(core::Dtype::Float32);
    t::geometry::Image target_intensity_curr =
            target.color_.RGBToGray().To(core::Dtype::Float32);

    // Create image pyramid
    for (int64_t i = 0; i < n_levels; ++i) {
        source_depth[n_levels - 1 - i] = source_depth_curr.AsTensor().Clone();
        target_depth[n_levels - 1 - i] = target_depth_curr.AsTensor().Clone();

        source_intensity[n_levels - 1 - i] =
                source_intensity_curr.AsTensor().Clone();
        target_intensity[n_levels - 1 - i] =
                target_intensity_curr.AsTensor().Clone();

        t::geometry::Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics, NAN);
        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();

        auto target_intensity_grad = target_intensity_curr.FilterSobel();
        target_intensity_dx[n_levels - 1 - i] =
                target_intensity_grad.first.AsTensor();
        target_intensity_dy[n_levels - 1 - i] =
                target_intensity_grad.second.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics.Clone();

        if (i != n_levels - 1) {
            source_depth_curr = PyrDownDepth(
                    source_depth_curr, params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = PyrDownDepth(
                    target_depth_curr, params.depth_outlier_trunc_ * 2, NAN);
            source_intensity_curr = source_intensity_curr.PyrDown();
            target_intensity_curr = target_intensity_curr.PyrDown();

            intrinsics /= 2;
            intrinsics[-1][-1] = 1;
        }
    }

    // Odometry
    OdometryResult result;
    double prev_rmse = 0.0;
    double prev_fitness = 1.0;
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < iterations[i]; ++iter) {
            result = ComputePoseIntensity(
                    source_depth[i], target_depth[i], source_intensity[i],
                    target_intensity[i], target_intensity_dx[i],
                    target_intensity_dy[i], source_vertex_maps[i],
                    intrinsic_matrices[i], trans, params.depth_outlier_trunc_,
                    params.intensity_huber_delta_);
            core::Tensor delta_source_to_target = result.transformation_;
            trans = delta_source_to_target.Matmul(trans);

            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, result.inlier_rmse_, result.fitness_);
            if (std::abs(prev_fitness - result.fitness_) / prev_fitness <
                        criteria.relative_fitness_ &&
                std::abs(prev_rmse - result.inlier_rmse_) / prev_rmse <
                        criteria.relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            prev_rmse = result.inlier_rmse_;
            prev_fitness = result.fitness_;
        }
    }

    return OdometryResult(trans, result.inlier_rmse_, result.fitness_);
}

OdometryResult RGBDOdometryMultiScaleHybrid(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        core::Tensor& intrinsics,
        core::Tensor& trans,
        float depth_scale,
        float depth_max,
        const std::vector<int>& iterations,
        const OdometryLossParams& params,
        const OdometryConvergenceCriteria& criteria) {
    int64_t n_levels = int64_t(iterations.size());
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

    t::geometry::Image source_depth_curr(source.depth_);
    t::geometry::Image target_depth_curr(target.depth_);

    t::geometry::Image source_intensity_curr =
            source.color_.RGBToGray().To(core::Dtype::Float32);
    t::geometry::Image target_intensity_curr =
            target.color_.RGBToGray().To(core::Dtype::Float32);

    // Create image pyramid
    for (int64_t i = 0; i < n_levels; ++i) {
        source_depth[n_levels - 1 - i] = source_depth_curr.AsTensor().Clone();
        target_depth[n_levels - 1 - i] = target_depth_curr.AsTensor().Clone();

        source_intensity[n_levels - 1 - i] =
                source_intensity_curr.AsTensor().Clone();
        target_intensity[n_levels - 1 - i] =
                target_intensity_curr.AsTensor().Clone();

        t::geometry::Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics, NAN);
        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();

        auto target_intensity_grad = target_intensity_curr.FilterSobel();
        target_intensity_dx[n_levels - 1 - i] =
                target_intensity_grad.first.AsTensor();
        target_intensity_dy[n_levels - 1 - i] =
                target_intensity_grad.second.AsTensor();

        auto target_depth_grad = target_depth_curr.FilterSobel();
        target_depth_dx[n_levels - 1 - i] = target_depth_grad.first.AsTensor();
        target_depth_dy[n_levels - 1 - i] = target_depth_grad.second.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics.Clone();

        if (i != n_levels - 1) {
            source_depth_curr = PyrDownDepth(
                    source_depth_curr, params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = PyrDownDepth(
                    target_depth_curr, params.depth_outlier_trunc_ * 2, NAN);
            source_intensity_curr = source_intensity_curr.PyrDown();
            target_intensity_curr = target_intensity_curr.PyrDown();

            intrinsics /= 2;
            intrinsics[-1][-1] = 1;
        }
    }

    // Odometry
    OdometryResult result;
    double prev_rmse = 0.0;
    double prev_fitness = 1.0;
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < iterations[i]; ++iter) {
            result = ComputePoseHybrid(
                    source_depth[i], target_depth[i], source_intensity[i],
                    target_intensity[i], target_depth_dx[i], target_depth_dy[i],
                    target_intensity_dx[i], target_intensity_dy[i],
                    source_vertex_maps[i], intrinsic_matrices[i], trans,
                    params.depth_outlier_trunc_, params.depth_huber_delta_,
                    params.intensity_huber_delta_);
            core::Tensor delta_source_to_target = result.transformation_;
            trans = delta_source_to_target.Matmul(trans);

            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, result.inlier_rmse_, result.fitness_);
            if (std::abs(prev_fitness - result.fitness_) / prev_fitness <
                        criteria.relative_fitness_ &&
                std::abs(prev_rmse - result.inlier_rmse_) / prev_rmse <
                        criteria.relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            prev_rmse = result.inlier_rmse_;
            prev_fitness = result.fitness_;
        }
    }

    return OdometryResult(trans, result.inlier_rmse_, result.fitness_);
}

OdometryResult ComputePosePointToPlane(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        float depth_outlier_trunc,
        float depth_huber_delta) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputePosePointToPlane(
            source_vertex_map, target_vertex_map, target_normal_map, intrinsics,
            init_source_to_target, se3_delta, inlier_residual, inlier_count,
            depth_outlier_trunc, depth_huber_delta);

    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape()[0] *
                                          source_vertex_map.GetShape()[1]));
}

OdometryResult ComputePoseIntensity(const core::Tensor& source_depth,
                                    const core::Tensor& target_depth,
                                    const core::Tensor& source_intensity,
                                    const core::Tensor& target_intensity,
                                    const core::Tensor& target_intensity_dx,
                                    const core::Tensor& target_intensity_dy,
                                    const core::Tensor& source_vertex_map,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    float depth_outlier_trunc,
                                    float intensity_huber_delta) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputePoseIntensity(
            source_depth, target_depth, source_intensity, target_intensity,
            target_intensity_dx, target_intensity_dy, source_vertex_map,
            intrinsics, init_source_to_target, se3_delta, inlier_residual,
            inlier_count, depth_outlier_trunc, intensity_huber_delta);

    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape()[0] *
                                          source_vertex_map.GetShape()[1]));
}

OdometryResult ComputePoseHybrid(const core::Tensor& source_depth,
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
                                 float depth_outlier_trunc,
                                 float depth_huber_delta,
                                 float intensity_huber_delta) {
    // Delta target_to_source on host.
    core::Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputePoseHybrid(
            source_depth, target_depth, source_intensity, target_intensity,
            target_depth_dx, target_depth_dy, target_intensity_dx,
            target_intensity_dy, source_vertex_map, intrinsics,
            init_source_to_target, se3_delta, inlier_residual, inlier_count,
            depth_outlier_trunc, depth_huber_delta, intensity_huber_delta);

    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape()[0] *
                                          source_vertex_map.GetShape()[1]));
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
