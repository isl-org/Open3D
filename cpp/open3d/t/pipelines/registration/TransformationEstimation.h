// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace t {
namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

namespace {

// Minimum time period (sec) between two sequential scans for Doppler ICP.
constexpr double kMinTimePeriod{1e-3};

}  // namespace

enum class TransformationEstimationType {
    Unspecified = 0,
    PointToPoint = 1,
    PointToPlane = 2,
    ColoredICP = 3,
    DopplerICP = 4,
};

/// \class TransformationEstimation
///
/// Base class that estimates a transformation between two point clouds
/// The virtual function ComputeTransformation() must be implemented in
/// subclasses.
class TransformationEstimation {
public:
    /// \brief Default Constructor.
    TransformationEstimation() {}
    virtual ~TransformationEstimation() {}

public:
    virtual TransformationEstimationType GetTransformationEstimationType()
            const = 0;

    /// Compute RMSE between source and target points cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud. (Float32 or Float64 type).
    /// \param target Target point cloud. (Float32 or Float64 type).
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    virtual double ComputeRMSE(const geometry::PointCloud &source,
                               const geometry::PointCloud &target,
                               const core::Tensor &correspondences) const = 0;
    /// Compute transformation from source to target point cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud. (Float32 or Float64 type).
    /// \param target Target point cloud. (Float32 or Float64 type).
    /// \param correspondences tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    /// \param current_transform The current pose estimate of ICP.
    /// \param iteration The current iteration number of the ICP algorithm.
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    virtual core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences,
            const core::Tensor &current_transform =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            const std::size_t iteration = 0) const = 0;
};

/// \class TransformationEstimationPointToPoint
///
/// Class to estimate a transformation matrix tensor of shape {4, 4}, dtype
/// Float64, on CPU device for point to point distance.
class TransformationEstimationPointToPoint : public TransformationEstimation {
public:
    // TODO: support with_scaling.
    TransformationEstimationPointToPoint() {}
    ~TransformationEstimationPointToPoint() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    /// \brief Computes RMSE (double) for PointToPoint method, between two
    /// pointclouds, given correspondences.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type).
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const core::Tensor &correspondences) const override;

    /// \brief Estimates the transformation matrix for PointToPoint method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type).
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    /// \param current_transform The current pose estimate of ICP.
    /// \param iteration The current iteration number of the ICP algorithm.
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences,
            const core::Tensor &current_transform =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            const std::size_t iteration = 0) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// \class TransformationEstimationPointToPlane
///
/// Class to estimate a transformation matrix tensor of shape {4, 4}, dtype
/// Float64, on CPU device for point to plane distance.
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    /// \brief Default constructor.
    TransformationEstimationPointToPlane() {}
    ~TransformationEstimationPointToPlane() override {}

    /// \brief Constructor that takes as input a RobustKernel
    ///
    /// \param kernel Any of the implemented statistical robust kernel for
    /// outlier rejection.
    explicit TransformationEstimationPointToPlane(const RobustKernel &kernel)
        : kernel_(kernel) {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };

    /// \brief Computes RMSE (double) for PointToPlane method, between two
    /// pointclouds, given correspondences.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals of the same shape and dtype as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const core::Tensor &correspondences) const override;

    /// \brief Estimates the transformation matrix for PointToPlane method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals of the same shape and dtype as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    /// \param current_transform The current pose estimate of ICP.
    /// \param iteration The current iteration number of the ICP algorithm.
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences,
            const core::Tensor &current_transform =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            const std::size_t iteration = 0) const override;

public:
    /// RobustKernel for outlier rejection.
    RobustKernel kernel_ = RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0);

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

/// \class TransformationEstimationForColoredICP
///
/// This is implementation of following paper
/// J. Park, Q.-Y. Zhou, V. Koltun,
/// Colored Point Cloud Registration Revisited, ICCV 2017.
///
/// Class to estimate a transformation matrix tensor of shape {4, 4}, dtype
/// Float64, on CPU device for colored-icp method.
class TransformationEstimationForColoredICP : public TransformationEstimation {
public:
    ~TransformationEstimationForColoredICP() override{};

    /// \brief Constructor.
    ///
    /// \param lamda_geometric  `λ ∈ [0,1]` in the overall energy `λEG +
    /// (1−λ)EC`. Refer the documentation of Colored-ICP for more information.
    /// \param kernel (optional) Any of the implemented statistical robust
    /// kernel for outlier rejection.
    explicit TransformationEstimationForColoredICP(
            double lambda_geometric = 0.968,
            const RobustKernel &kernel =
                    RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0))
        : lambda_geometric_(lambda_geometric), kernel_(kernel) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0) {
            lambda_geometric_ = 0.968;
        }
    }

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };

public:
    /// \brief Computes RMSE (double) for ColoredICP method, between two
    /// pointclouds, given correspondences.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals, colors and color_gradients of the same shape and dtype
    /// as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const core::Tensor &correspondences) const override;

    /// \brief Estimates the transformation matrix for ColoredICP method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals, colors and color_gradients of the same shape and dtype
    /// as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    /// \param current_transform The current pose estimate of ICP.
    /// \param iteration The current iteration number of the ICP algorithm.
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences,
            const core::Tensor &current_transform =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            const std::size_t iteration = 0) const override;

public:
    double lambda_geometric_ = 0.968;
    /// RobustKernel for outlier rejection.
    RobustKernel kernel_ = RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0);

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::ColoredICP;
};

/// \class TransformationEstimationForDopplerICP
///
/// This is the implementation of the following paper:
/// B. Hexsel, H. Vhavle, Y. Chen,
/// DICP: Doppler Iterative Closest Point Algorithm, RSS 2022.
///
/// Class to estimate a transformation matrix tensor of shape {4, 4}, dtype
/// Float64, on CPU device for DopplerICP method.
class TransformationEstimationForDopplerICP : public TransformationEstimation {
public:
    ~TransformationEstimationForDopplerICP() override{};

    /// \brief Constructor.
    ///
    /// \param period Time period (in seconds) between the source and the target
    /// point clouds. Default value: 0.1.
    /// \param lambda_doppler `λ ∈ [0, 1]` in the overall energy `(1−λ)EG +
    /// λED`. Refer the documentation of DopplerICP for more information.
    /// \param reject_dynamic_outliers Whether or not to reject dynamic point
    /// outlier correspondences.
    /// \param doppler_outlier_threshold Correspondences with Doppler error
    /// greater than this threshold are rejected from optimization.
    /// \param outlier_rejection_min_iteration Number of iterations of ICP after
    /// which outlier rejection is enabled.
    /// \param geometric_robust_loss_min_iteration Number of iterations of ICP
    /// after which robust loss for geometric term kicks in.
    /// \param doppler_robust_loss_min_iteration Number of iterations of ICP
    /// after which robust loss for Doppler term kicks in.
    /// \param geometric_kernel (optional) Any of the implemented statistical
    /// robust kernel for outlier rejection for the geometric term.
    /// \param doppler_kernel (optional) Any of the implemented statistical
    /// robust kernel for outlier rejection for the Doppler term.
    /// \param transform_vehicle_to_sensor The 4x4 extrinsic transformation
    /// matrix between the vehicle and the sensor frames. Defaults to identity
    /// transform.
    explicit TransformationEstimationForDopplerICP(
            const double period = 0.1,
            const double lambda_doppler = 0.01,
            const bool reject_dynamic_outliers = false,
            const double doppler_outlier_threshold = 2.0,
            const std::size_t outlier_rejection_min_iteration = 2,
            const std::size_t geometric_robust_loss_min_iteration = 0,
            const std::size_t doppler_robust_loss_min_iteration = 2,
            const RobustKernel &geometric_kernel =
                    RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0),
            const RobustKernel &doppler_kernel =
                    RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0),
            const core::Tensor &transform_vehicle_to_sensor =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")))
        : period_(period),
          lambda_doppler_(lambda_doppler),
          reject_dynamic_outliers_(reject_dynamic_outliers),
          doppler_outlier_threshold_(doppler_outlier_threshold),
          outlier_rejection_min_iteration_(outlier_rejection_min_iteration),
          geometric_robust_loss_min_iteration_(
                  geometric_robust_loss_min_iteration),
          doppler_robust_loss_min_iteration_(doppler_robust_loss_min_iteration),
          geometric_kernel_(geometric_kernel),
          doppler_kernel_(doppler_kernel),
          transform_vehicle_to_sensor_(transform_vehicle_to_sensor) {
        core::AssertTensorShape(transform_vehicle_to_sensor, {4, 4});

        if (std::abs(period) < kMinTimePeriod) {
            utility::LogError("Time period too small.");
        }

        if (lambda_doppler_ < 0 || lambda_doppler_ > 1.0) {
            lambda_doppler_ = 0.01;
        }
    }

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };

public:
    /// \brief Computes RMSE (double) for DopplerICP method, between two
    /// pointclouds, given correspondences.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals, directions, and Doppler of the same shape and dtype
    /// as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const core::Tensor &correspondences) const override;

    /// \brief Estimates the transformation matrix for DopplerICP method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud. (Float32 or Float64 type).
    /// \param target Target pointcloud. (Float32 or Float64 type). It must
    /// contain normals, directions, and Doppler of the same shape and dtype
    /// as the positions.
    /// \param correspondences Tensor of type Int64 containing indices of
    /// corresponding target points, where the value is the target index and the
    /// index of the value itself is the source index. It contains -1 as value
    /// at index with no correspondence.
    /// \param current_transform The current pose estimate of ICP.
    /// \param iteration The current iteration number of the ICP algorithm.
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences,
            const core::Tensor &current_transform =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            const std::size_t iteration = 0) const override;

public:
    /// Time period (in seconds) between the source and the target point clouds.
    double period_{0.1};
    /// Factor that weighs the Doppler residual term in DICP objective.
    double lambda_doppler_{0.01};
    /// Whether or not to prune dynamic point outlier correspondences.
    bool reject_dynamic_outliers_{false};
    /// Correspondences with Doppler error greater than this threshold are
    /// rejected from optimization.
    double doppler_outlier_threshold_{2.0};
    /// Number of iterations of ICP after which outlier rejection is enabled.
    std::size_t outlier_rejection_min_iteration_{2};
    /// Number of iterations of ICP after which robust loss kicks in.
    std::size_t geometric_robust_loss_min_iteration_{0};
    std::size_t doppler_robust_loss_min_iteration_{2};
    /// RobustKernel for outlier rejection.
    RobustKernel geometric_kernel_ =
            RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0);
    RobustKernel doppler_kernel_ =
            RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0);
    /// The 4x4 extrinsic transformation matrix between the vehicle and the
    /// sensor frames.
    core::Tensor transform_vehicle_to_sensor_ =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::DopplerICP;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
