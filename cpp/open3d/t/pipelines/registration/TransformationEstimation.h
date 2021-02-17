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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {

namespace t {
namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

/// CorrespondenceSet is a pair of tensor, where first tensor
/// [correspondence_select_bool_] is a {N,1} bool tensor (N is the number of
/// query points), with value true for source points having good correspondence,
/// and false otherwise, and second [correspondence_set_] is a {C,1} shape
/// Float32 tensor (C is the number of good correspondences), where value at
/// [i, 1] is the corresponding index in the target, for query point [i, 1].
typedef std::pair<core::Tensor, core::Tensor> CorrespondenceSet;

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
                               CorrespondenceSet &corres) const = 0;
    /// Compute transformation from source to target point cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    virtual core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            CorrespondenceSet &corres) const = 0;
};

/// \class TransformationEstimationPointToPoint
///
/// Estimate a transformation for point to point distance.
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
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       CorrespondenceSet &corres) const override;
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            CorrespondenceSet &corres) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// \class TransformationEstimationPointToPlane
///
/// Class to estimate a transformation for point to plane distance.
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    /// \brief Default constructor.
    TransformationEstimationPointToPlane() {}
    ~TransformationEstimationPointToPlane() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       CorrespondenceSet &corres) const override;
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            CorrespondenceSet &corres) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
