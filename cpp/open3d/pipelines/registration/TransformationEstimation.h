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

#include <Eigen/Core>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open3d/pipelines/registration/RobustKernel.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

typedef std::vector<Eigen::Vector2i> CorrespondenceSet;

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
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    virtual double ComputeRMSE(const geometry::PointCloud &source,
                               const geometry::PointCloud &target,
                               const CorrespondenceSet &corres) const = 0;
    /// Compute transformation from source to target point cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    virtual Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const = 0;
};

/// \class TransformationEstimationPointToPoint
///
/// Estimate a transformation for point to point distance.
class TransformationEstimationPointToPoint : public TransformationEstimation {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param with_scaling Set to True to estimate scaling, False to force
    /// scaling to be 1.
    TransformationEstimationPointToPoint(bool with_scaling = false)
        : with_scaling_(with_scaling) {}
    ~TransformationEstimationPointToPoint() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    /// \brief Set to True to estimate scaling, False to force scaling to be 1.
    ///
    /// The homogeneous transformation is given by\n
    /// T = [ cR t]\n
    ///    [0   1]\n
    /// Sets 𝑐=1 if with_scaling is False.
    bool with_scaling_ = false;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// \class TransformationEstimationPointToPlane
///
/// Class to estimate a transformation for point to plane distance.
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    /// \brief Default Constructor.
    TransformationEstimationPointToPlane() {}
    ~TransformationEstimationPointToPlane() override {}

    /// \brief Constructor that takes as input a RobustKernel \params kernel Any
    /// of the implemented statistical robust kernel for outlier rejection.
    explicit TransformationEstimationPointToPlane(
            std::shared_ptr<RobustKernel> kernel)
        : kernel_(std::move(kernel)) {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
