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

using core::Tensor;
using t::geometry::Image;
using t::geometry::RGBDImage;

OdometryResult RGBDOdometryMultiScalePointToPlane(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params);

OdometryResult RGBDOdometryMultiScaleIntensity(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsic,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params);

OdometryResult RGBDOdometryMultiScaleHybrid(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params);

OdometryResult RGBDOdometryMultiScale(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& init_source_to_target,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const Method method,
        const OdometryLossParams& params) {
    // TODO (wei): more device check
    const core::Device device = source.depth_.GetDevice();
    core::AssertTensorDevice(target.depth_.AsTensor(), device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    // 4x4 transformations are always float64 and stay on CPU.
    const core::Device host("CPU:0");
    const Tensor intrinsics_d = intrinsics.To(host, core::Float64).Clone();
    const Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Clone();

    Image source_depth = source.depth_;
    Image target_depth = target.depth_;

    Image source_depth_processed =
            source_depth.ClipTransform(depth_scale, 0, depth_max, NAN);
    Image target_depth_processed =
            target_depth.ClipTransform(depth_scale, 0, depth_max, NAN);

    RGBDImage source_processed(source.color_, source_depth_processed);
    RGBDImage target_processed(target.color_, target_depth_processed);

    if (method == Method::PointToPlane) {
        return RGBDOdometryMultiScalePointToPlane(
                source_processed, target_processed, intrinsics_d, trans_d,
                depth_scale, depth_max, criteria, params);
    } else if (method == Method::Intensity) {
        return RGBDOdometryMultiScaleIntensity(
                source_processed, target_processed, intrinsics_d, trans_d,
                depth_scale, depth_max, criteria, params);
    } else if (method == Method::Hybrid) {
        return RGBDOdometryMultiScaleHybrid(source_processed, target_processed,
                                            intrinsics_d, trans_d, depth_scale,
                                            depth_max, criteria, params);
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return OdometryResult(trans_d);
}

OdometryResult RGBDOdometryMultiScalePointToPlane(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params) {
    int64_t n_levels = int64_t(criteria.size());
    std::vector<Tensor> source_vertex_maps(n_levels);
    std::vector<Tensor> target_vertex_maps(n_levels);
    std::vector<Tensor> target_normal_maps(n_levels);
    std::vector<Tensor> intrinsic_matrices(n_levels);

    Image source_depth_curr = source.depth_;
    Image target_depth_curr = target.depth_;

    Tensor intrinsics_pyr = intrinsics;

    // Create image pyramid.
    for (int64_t i = 0; i < n_levels; ++i) {
        Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics_pyr, NAN);
        Image target_vertex_map =
                target_depth_curr.CreateVertexMap(intrinsics_pyr, NAN);

        Image target_depth_curr_smooth =
                target_depth_curr.FilterBilateral(5, 5, 10);
        Image target_vertex_map_smooth =
                target_depth_curr_smooth.CreateVertexMap(intrinsics_pyr, NAN);
        Image target_normal_map = target_vertex_map_smooth.CreateNormalMap(NAN);

        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();
        target_vertex_maps[n_levels - 1 - i] = target_vertex_map.AsTensor();
        target_normal_maps[n_levels - 1 - i] = target_normal_map.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics_pyr.Clone();

        if (i != n_levels - 1) {
            source_depth_curr = source_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = target_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);

            intrinsics_pyr /= 2;
            intrinsics_pyr[-1][-1] = 1;
        }
    }

    OdometryResult result(trans, /*prev rmse*/ 0.0, /*prev fitness*/ 1.0);
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < criteria[i].max_iteration_; ++iter) {
            auto delta_result = ComputeOdometryResultPointToPlane(
                    source_vertex_maps[i], target_vertex_maps[i],
                    target_normal_maps[i], intrinsic_matrices[i],
                    result.transformation_, params.depth_outlier_trunc_,
                    params.depth_huber_delta_);
            result.transformation_ =
                    delta_result.transformation_.Matmul(result.transformation_);
            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, delta_result.inlier_rmse_,
                              delta_result.fitness_);

            if (std::abs(result.fitness_ - delta_result.fitness_) /
                                result.fitness_ <
                        criteria[i].relative_fitness_ &&
                std::abs(result.inlier_rmse_ - delta_result.inlier_rmse_) /
                                result.inlier_rmse_ <
                        criteria[i].relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            result.inlier_rmse_ = delta_result.inlier_rmse_;
            result.fitness_ = delta_result.fitness_;
        }
    }

    return result;
}

OdometryResult RGBDOdometryMultiScaleIntensity(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params) {
    int64_t n_levels = int64_t(criteria.size());
    std::vector<Tensor> source_intensity(n_levels);
    std::vector<Tensor> target_intensity(n_levels);

    std::vector<Tensor> source_depth(n_levels);
    std::vector<Tensor> target_depth(n_levels);
    std::vector<Tensor> target_intensity_dx(n_levels);
    std::vector<Tensor> target_intensity_dy(n_levels);

    std::vector<Tensor> source_vertex_maps(n_levels);

    std::vector<Tensor> intrinsic_matrices(n_levels);

    Image source_depth_curr = source.depth_;
    Image target_depth_curr = target.depth_;

    Image source_intensity_curr = source.color_.RGBToGray().To(core::Float32);
    Image target_intensity_curr = target.color_.RGBToGray().To(core::Float32);

    Tensor intrinsics_pyr = intrinsics;

    // Create image pyramid
    for (int64_t i = 0; i < n_levels; ++i) {
        source_depth[n_levels - 1 - i] = source_depth_curr.AsTensor().Clone();
        target_depth[n_levels - 1 - i] = target_depth_curr.AsTensor().Clone();

        source_intensity[n_levels - 1 - i] =
                source_intensity_curr.AsTensor().Clone();
        target_intensity[n_levels - 1 - i] =
                target_intensity_curr.AsTensor().Clone();

        Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics_pyr, NAN);
        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();

        auto target_intensity_grad = target_intensity_curr.FilterSobel();
        target_intensity_dx[n_levels - 1 - i] =
                target_intensity_grad.first.AsTensor();
        target_intensity_dy[n_levels - 1 - i] =
                target_intensity_grad.second.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics_pyr.Clone();

        if (i != n_levels - 1) {
            source_depth_curr = source_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = target_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);
            source_intensity_curr = source_intensity_curr.PyrDown();
            target_intensity_curr = target_intensity_curr.PyrDown();

            intrinsics_pyr /= 2;
            intrinsics_pyr[-1][-1] = 1;
        }
    }

    // Odometry
    OdometryResult result(trans, /*prev rmse*/ 0.0, /*prev fitness*/ 1.0);
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < criteria[i].max_iteration_; ++iter) {
            auto delta_result = ComputeOdometryResultIntensity(
                    source_depth[i], target_depth[i], source_intensity[i],
                    target_intensity[i], target_intensity_dx[i],
                    target_intensity_dy[i], source_vertex_maps[i],
                    intrinsic_matrices[i], result.transformation_,
                    params.depth_outlier_trunc_, params.intensity_huber_delta_);
            result.transformation_ =
                    delta_result.transformation_.Matmul(result.transformation_);
            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, delta_result.inlier_rmse_,
                              delta_result.fitness_);

            if (std::abs(result.fitness_ - delta_result.fitness_) /
                                result.fitness_ <
                        criteria[i].relative_fitness_ &&
                std::abs(result.inlier_rmse_ - delta_result.inlier_rmse_) /
                                result.inlier_rmse_ <
                        criteria[i].relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            result.inlier_rmse_ = delta_result.inlier_rmse_;
            result.fitness_ = delta_result.fitness_;
        }
    }

    return result;
}

OdometryResult RGBDOdometryMultiScaleHybrid(
        const RGBDImage& source,
        const RGBDImage& target,
        const Tensor& intrinsics,
        const Tensor& trans,
        const float depth_scale,
        const float depth_max,
        const std::vector<OdometryConvergenceCriteria>& criteria,
        const OdometryLossParams& params) {
    int64_t n_levels = int64_t(criteria.size());
    std::vector<Tensor> source_intensity(n_levels);
    std::vector<Tensor> target_intensity(n_levels);

    std::vector<Tensor> source_depth(n_levels);
    std::vector<Tensor> target_depth(n_levels);
    std::vector<Tensor> target_intensity_dx(n_levels);
    std::vector<Tensor> target_intensity_dy(n_levels);

    std::vector<Tensor> target_depth_dx(n_levels);
    std::vector<Tensor> target_depth_dy(n_levels);

    std::vector<Tensor> source_vertex_maps(n_levels);

    std::vector<Tensor> intrinsic_matrices(n_levels);

    Image source_depth_curr(source.depth_);
    Image target_depth_curr(target.depth_);

    Image source_intensity_curr = source.color_.RGBToGray().To(core::Float32);
    Image target_intensity_curr = target.color_.RGBToGray().To(core::Float32);

    Tensor intrinsics_pyr = intrinsics;
    // Create image pyramid
    for (int64_t i = 0; i < n_levels; ++i) {
        source_depth[n_levels - 1 - i] = source_depth_curr.AsTensor().Clone();
        target_depth[n_levels - 1 - i] = target_depth_curr.AsTensor().Clone();

        source_intensity[n_levels - 1 - i] =
                source_intensity_curr.AsTensor().Clone();
        target_intensity[n_levels - 1 - i] =
                target_intensity_curr.AsTensor().Clone();

        Image source_vertex_map =
                source_depth_curr.CreateVertexMap(intrinsics_pyr, NAN);
        source_vertex_maps[n_levels - 1 - i] = source_vertex_map.AsTensor();

        auto target_intensity_grad = target_intensity_curr.FilterSobel();
        target_intensity_dx[n_levels - 1 - i] =
                target_intensity_grad.first.AsTensor();
        target_intensity_dy[n_levels - 1 - i] =
                target_intensity_grad.second.AsTensor();

        auto target_depth_grad = target_depth_curr.FilterSobel();
        target_depth_dx[n_levels - 1 - i] = target_depth_grad.first.AsTensor();
        target_depth_dy[n_levels - 1 - i] = target_depth_grad.second.AsTensor();

        intrinsic_matrices[n_levels - 1 - i] = intrinsics_pyr.Clone();
        if (i != n_levels - 1) {
            source_depth_curr = source_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);
            target_depth_curr = target_depth_curr.PyrDownDepth(
                    params.depth_outlier_trunc_ * 2, NAN);
            source_intensity_curr = source_intensity_curr.PyrDown();
            target_intensity_curr = target_intensity_curr.PyrDown();

            intrinsics_pyr /= 2;
            intrinsics_pyr[-1][-1] = 1;
        }
    }

    // Odometry
    OdometryResult result(trans, /*prev rmse*/ 0.0, /*prev fitness*/ 1.0);
    for (int64_t i = 0; i < n_levels; ++i) {
        for (int iter = 0; iter < criteria[i].max_iteration_; ++iter) {
            auto delta_result = ComputeOdometryResultHybrid(
                    source_depth[i], target_depth[i], source_intensity[i],
                    target_intensity[i], target_depth_dx[i], target_depth_dy[i],
                    target_intensity_dx[i], target_intensity_dy[i],
                    source_vertex_maps[i], intrinsic_matrices[i],
                    result.transformation_, params.depth_outlier_trunc_,
                    params.depth_huber_delta_, params.intensity_huber_delta_);
            result.transformation_ =
                    delta_result.transformation_.Matmul(result.transformation_);
            utility::LogDebug("level {}, iter {}: rmse = {}, fitness = {}", i,
                              iter, delta_result.inlier_rmse_,
                              delta_result.fitness_);

            if (std::abs(result.fitness_ - delta_result.fitness_) /
                                result.fitness_ <
                        criteria[i].relative_fitness_ &&
                std::abs(result.inlier_rmse_ - delta_result.inlier_rmse_) /
                                result.inlier_rmse_ <
                        criteria[i].relative_rmse_) {
                utility::LogDebug("Early exit at level {}, iter {}", i, iter);
                break;
            }
            result.inlier_rmse_ = delta_result.inlier_rmse_;
            result.fitness_ = delta_result.fitness_;
        }
    }

    return result;
}

OdometryResult ComputeOdometryResultPointToPlane(
        const Tensor& source_vertex_map,
        const Tensor& target_vertex_map,
        const Tensor& target_normal_map,
        const Tensor& intrinsics,
        const Tensor& init_source_to_target,
        const float depth_outlier_trunc,
        const float depth_huber_delta) {
    // Delta target_to_source on host.
    Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputeOdometryResultPointToPlane(
            source_vertex_map, target_vertex_map, target_normal_map, intrinsics,
            init_source_to_target, se3_delta, inlier_residual, inlier_count,
            depth_outlier_trunc, depth_huber_delta);
    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }
    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
}

OdometryResult ComputeOdometryResultIntensity(
        const Tensor& source_depth,
        const Tensor& target_depth,
        const Tensor& source_intensity,
        const Tensor& target_intensity,
        const Tensor& target_intensity_dx,
        const Tensor& target_intensity_dy,
        const Tensor& source_vertex_map,
        const Tensor& intrinsics,
        const Tensor& init_source_to_target,
        const float depth_outlier_trunc,
        const float intensity_huber_delta) {
    // Delta target_to_source on host.
    Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputeOdometryResultIntensity(
            source_depth, target_depth, source_intensity, target_intensity,
            target_intensity_dx, target_intensity_dy, source_vertex_map,
            intrinsics, init_source_to_target, se3_delta, inlier_residual,
            inlier_count, depth_outlier_trunc, intensity_huber_delta);
    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }
    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
}

OdometryResult ComputeOdometryResultHybrid(const Tensor& source_depth,
                                           const Tensor& target_depth,
                                           const Tensor& source_intensity,
                                           const Tensor& target_intensity,
                                           const Tensor& target_depth_dx,
                                           const Tensor& target_depth_dy,
                                           const Tensor& target_intensity_dx,
                                           const Tensor& target_intensity_dy,
                                           const Tensor& source_vertex_map,
                                           const Tensor& intrinsics,
                                           const Tensor& init_source_to_target,
                                           const float depth_outlier_trunc,
                                           const float depth_huber_delta,
                                           const float intensity_huber_delta) {
    // Delta target_to_source on host.
    Tensor se3_delta;
    float inlier_residual;
    int inlier_count;
    kernel::odometry::ComputeOdometryResultHybrid(
            source_depth, target_depth, source_intensity, target_intensity,
            target_depth_dx, target_depth_dy, target_intensity_dx,
            target_intensity_dy, source_vertex_map, intrinsics,
            init_source_to_target, se3_delta, inlier_residual, inlier_count,
            depth_outlier_trunc, depth_huber_delta, intensity_huber_delta);
    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }
    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
