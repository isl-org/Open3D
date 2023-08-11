// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

namespace open3d {

namespace t {
namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

enum class TransformationEstimationType {
    Unspecified = 0,
    PointToPoint = 1,
    PointToPlane = 2,
    ColoredICP = 3,
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
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    virtual core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences) const = 0;
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
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences) const override;

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
    /// \return transformation between source to target, a tensor
    /// of shape {4, 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences) const override;

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
    /// \return transformation between source to target, a tensor of shape {4,
    /// 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const core::Tensor &correspondences) const override;

public:
    double lambda_geometric_ = 0.968;
    /// RobustKernel for outlier rejection.
    RobustKernel kernel_ = RobustKernel(RobustKernelMethod::L2Loss, 1.0, 1.0);

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::ColoredICP;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
