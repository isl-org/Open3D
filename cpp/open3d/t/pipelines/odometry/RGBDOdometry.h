// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file RGBDOdometry.h
/// All the 4x4 transformation in this file, from params to returns, are
/// Float64. Only convert to Float32 in kernel calls.

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

enum class Method {
    PointToPlane,  // Implemented and commented in
                   // ComputeOdometryResultPointToPlane
    Intensity,  // Implemented and commented in ComputeOdometryResultIntensity
    Hybrid,     // Implemented and commented in ComputeOdometryResultHybrid
};

class OdometryConvergenceCriteria {
public:
    /// \brief Constructor for the convergence criteria, where we stop
    /// iterations once the criteria are met.
    ///
    /// \param max_iteration Maximum iteration before iteration stops.
    /// \param relative_rmse Relative rmse threshold where we stop iterations
    /// when \f$ |rmse_{i+1} - rmse_i|/rmse_i < relative rmse\f$.
    /// \param relative_fitness Relative fitness threshold where we stop
    /// iterations when \f$ |fitness_{i+1} - fitness_i|/fitness_i < relative
    /// fitness\f$
    OdometryConvergenceCriteria(int max_iteration,
                                double relative_rmse = 1e-6,
                                double relative_fitness = 1e-6)
        : max_iteration_(max_iteration),
          relative_rmse_(relative_rmse),
          relative_fitness_(relative_fitness) {}

public:
    /// Maximum iteration before iteration stops.
    int max_iteration_;
    /// If relative change (difference) of inliner RMSE score is lower than
    /// `relative_rmse`, the iteration stops.
    double relative_rmse_;
    /// If relative change (difference) of fitness score is lower than
    /// `relative_fitness`, the iteration stops.
    double relative_fitness_;
};

class OdometryResult {
public:
    /// \brief Constructor for the odometry result.
    ///
    /// \param transformation The estimated transformation matrix of dtype
    /// Float64 on CPU device.
    /// \param inlier_rmse RMSE of the inliers.
    /// \param fitness Ratio between #inliers and #pixels.
    OdometryResult(const core::Tensor& transformation = core::Tensor::Eye(
                           4, core::Float64, core::Device("CPU:0")),
                   double inlier_rmse = 0.0,
                   double fitness = 0.0)
        : transformation_(transformation),
          inlier_rmse_(inlier_rmse),
          fitness_(fitness) {}

    ~OdometryResult() {}

public:
    /// The estimated transformation matrix of dtype Float64 on CPU device.
    core::Tensor transformation_;
    /// RMSE of all inlier. Lower is better.
    double inlier_rmse_;
    /// The overlapping area (# of inlier correspondences / # of points
    /// in target). Higher is better.
    double fitness_;
};

class OdometryLossParams {
public:
    /// \brief Constructor for the odometry loss function.
    ///
    /// \param depth_outlier_trunc Threshold to filter outlier associations
    /// where two depths differ significantly.
    /// \param depth_huber_delta Huber norm parameter applied to depth loss (for
    /// PointToPlane and Hybrid).
    /// \param intensity_huber_delta Huber norm parameter applied to intensity
    /// loss (for Intensity and Hybrid).
    OdometryLossParams(float depth_outlier_trunc = 0.07,
                       float depth_huber_delta = 0.05,
                       float intensity_huber_delta = 0.1)
        : depth_outlier_trunc_(depth_outlier_trunc),
          depth_huber_delta_(depth_huber_delta),
          intensity_huber_delta_(intensity_huber_delta) {
        if (depth_outlier_trunc_ < 0) {
            utility::LogWarning(
                    "Depth outlier truncation < 0, outliers will be counted!");
        }
        if (depth_huber_delta_ >= depth_outlier_trunc_) {
            utility::LogWarning(
                    "Huber delta is greater than truncation, huber norm will "
                    "degenerate to L2 norm!");
        }
    }

public:
    /// Depth difference threshold used to filter projective associations.
    float depth_outlier_trunc_;
    float depth_huber_delta_;
    float intensity_huber_delta_;
};

/// \brief Create an RGBD image pyramid given the original source and target
/// RGBD images, and perform hierarchical odometry using specified \p
/// method.
/// Can be used for offline odometry where we do not expect to push performance
/// to the extreme and not reuse vertex/normal map computed before.
/// Input RGBD images hold a depth image (UInt16 or Float32) with a scale
/// factor and a color image (UInt8 x 3).
/// \param source Source RGBD image.
/// \param target Target RGBD image.
/// \param intrinsics (3, 3) intrinsic matrix for projection of core::Float64
/// on CPU.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target of core::Float64 on CPU.
/// \param depth_scale Converts depth pixel values to meters by dividing the
/// scale factor.
/// \param depth_max Max depth to truncate depth image with noisy measurements.
/// \param criteria_list Criteria used to define and terminate iterations. In
/// multiscale odometry the order is from coarse to fine. Inputting a vector of
/// iterations by default triggers the implicit conversion.
/// \param method Method used to apply RGBD odometry.
/// \param params Parameters used in loss function, including outlier rejection
/// threshold and Huber norm parameters.
/// \return odometry result, with (4, 4) optimized transformation matrix from
/// source to target, inlier ratio, and fitness.
OdometryResult RGBDOdometryMultiScale(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        const float depth_scale = 1000.0f,
        const float depth_max = 3.0f,
        const std::vector<OdometryConvergenceCriteria>& criteria_list = {10, 5,
                                                                         3},
        const Method method = Method::Hybrid,
        const OdometryLossParams& params = OdometryLossParams());

/// \brief Estimates the 4x4 rigid transformation T from source to target, with
/// inlier rmse and fitness.
/// Performs one iteration of RGBD odometry using loss function
/// \f$[(V_p - V_q)^T N_p]^2\f$, where
/// \f$ V_p \f$ denotes the vertex at pixel p in the source,
/// \f$ V_q \f$ denotes the vertex at pixel q in the target,
/// \f$ N_p \f$ denotes the normal at pixel p in the source.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// KinectFusion, ISMAR 2011
///
/// \param source_vertex_map (rows, cols, channels=3) Float32 source vertex
/// image obtained by CreateVertexMap before calling this function.
/// \param target_vertex_map (rows, cols, channels=3) Float32 target vertex
/// image obtained by CreateVertexMap before calling this function.
/// \param target_normal_map (rows, cols, channels=3) Float32 target normal
/// image obtained by CreateNormalMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_outlier_trunc Depth difference threshold used to filter
/// projective associations.
/// \param depth_huber_delta Huber norm parameter used in depth loss.
/// \return odometry result, with (4, 4) optimized transformation matrix from
/// source to target, inlier ratio, and fitness.
OdometryResult ComputeOdometryResultPointToPlane(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

/// \brief Estimates the 4x4 rigid transformation T from source to target, with
/// inlier rmse and fitness.
/// Performs one iteration of RGBD odometry using loss function
/// \f$(I_p - I_q)^2\f$, where
/// \f$ I_p \f$ denotes the intensity at pixel p in the source,
/// \f$ I_q \f$ denotes the intensity at pixel q in the target.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// Real-time visual odometry from dense RGB-D images, ICCV Workshops, 2011
///
/// \param source_depth (rows, cols, channels=1) Float32 source depth image
/// obtained by PreprocessDepth before calling this function.
/// \param target_depth (rows, cols, channels=1) Float32 target depth image
/// obtained by PreprocessDepth before calling this function.
/// \param source_intensity (rows, cols, channels=1) Float32 source intensity
/// image obtained by RGBToGray before calling this function.
/// \param target_intensity (rows, cols, channels=1) Float32 target intensity
/// image obtained by RGBToGray before calling this function.
/// \param target_intensity_dx (rows, cols, channels=1) Float32 target intensity
/// gradient image along x-axis obtained by FilterSobel before calling this
/// function.
/// \param target_intensity_dy (rows, cols, channels=1) Float32 target intensity
/// gradient image along y-axis obtained by FilterSobel before calling this
/// function.
/// \param source_vertex_map (rows, cols, channels=3) Float32 source vertex
/// image obtained by CreateVertexMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param  init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_outlier_trunc Depth difference threshold used to filter
/// projective associations.
/// \param intensity_huber_delta Huber norm parameter used in intensity loss.
/// \return odometry result, with(4, 4) optimized transformation matrix
/// from source to target, inlier ratio, and fitness.
OdometryResult ComputeOdometryResultIntensity(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        const float depth_outlier_trunc,
        const float intensity_huber_delta);

/// \brief Estimates the 4x4 rigid transformation T from source to target, with
/// inlier rmse and fitness.
/// Performs one iteration of RGBD odometry using loss function
/// \f$(I_p - I_q)^2 + \lambda(D_p - (D_q)')^2\f$, where
/// \f$ I_p \f$ denotes the intensity at pixel p in the source,
/// \f$ I_q \f$ denotes the intensity at pixel q in the target.
/// \f$ D_p \f$ denotes the depth pixel p in the source,
/// \f$ D_q \f$ denotes the depth pixel q in the target.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// Reference: J. Park, Q.Y. Zhou, and V. Koltun,
/// Colored Point Cloud Registration Revisited, ICCV, 2017.
///
/// \param source_depth (rows, cols, channels=1) Float32 source depth image
/// obtained by PreprocessDepth before calling this function.
/// \param target_depth (rows, cols, channels=1) Float32 target depth image
/// obtained by PreprocessDepth before calling this function.
/// \param source_intensity (rows, cols, channels=1) Float32 source intensity
/// image obtained by RGBToGray before calling this function.
/// \param target_intensity (rows, cols, channels=1) Float32 target intensity
/// image obtained by RGBToGray before calling this function.
/// \param target_depth_dx (rows, cols, channels=1) Float32 target depth
/// gradient image along x-axis obtained by FilterSobel before calling this
/// function.
/// \param target_depth_dy (rows, cols, channels=1) Float32 target depth
/// gradient image along y-axis obtained by FilterSobel before calling this
/// function.
/// \param target_intensity_dx (rows, cols, channels=1) Float32 target intensity
/// gradient image along x-axis obtained by FilterSobel before calling this
/// function.
/// \param target_intensity_dy (rows, cols, channels=1) Float32 target intensity
/// gradient image along y-axis obtained by FilterSobel before calling this
/// function.
/// \param source_vertex_map (rows, cols, channels=3) Float32 source vertex
/// image obtained by CreateVertexMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_outlier_trunc Depth difference threshold used to filter
/// projective associations.
/// \param depth_huber_delta Huber norm parameter used in depth loss.
/// \param intensity_huber_delta Huber norm parameter used in intensity loss.
/// \return odometry result, with(4, 4) optimized transformation matrix
/// from source to target, inlier ratio, and fitness.
OdometryResult ComputeOdometryResultHybrid(
        const core::Tensor& source_depth,
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
        const float depth_outlier_trunc,
        const float depth_huber_delta,
        const float intensity_huber_delta);

/// Estimates 6x6 information matrix from a pair of depth images.
/// The process is akin to information matrix creation for point clouds.
core::Tensor ComputeOdometryInformationMatrix(
        const geometry::Image& source_depth,
        const geometry::Image& target_depth,
        const core::Tensor& intrinsic,
        const core::Tensor& source_to_target,
        const float dist_thr,
        const float depth_scale = 1000.0f,
        const float depth_max = 3.0f);
}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
