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

/// \brief CorrespondenceSet: pair of 2x Int64 tensors of shape {C,},
/// where C is the number of good correspondences between source and
/// target pointcloud. The first tensor is the source indices, and
/// the second tensor is corresponding target indices. Such that,
/// source[correspondence.first] and target[correspondence.second]
/// is a correspondence pair.
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
    /// \param source Source point cloud of type Float32.
    /// \param target Target point cloud of type Float32.
    /// \param correspondences Correspondence set between source and target
    /// point cloud.
    virtual double ComputeRMSE(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &correspondences) const = 0;
    /// Compute transformation from source to target point cloud given
    /// correspondences.
    ///
    /// \param source Source point cloud of type Float32.
    /// \param target Target point cloud of type Float32.
    /// \param correspondences Correspondence set between source and target
    /// point cloud.
    /// \return transformation between source to target, a tensor of shape
    /// {4, 4}, type Float64 on CPU device.
    virtual core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &correspondences) const = 0;
};

/// \class TransformationEstimationPointToPoint
///
/// Class to estimate a transformation of shape {4, 4} and dtype Float64 for
/// point to point distance.
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
    /// pointclouds of type Float32, given CorrespondenceSet.
    ///
    /// \param source Source pointcloud of dtype Float32.
    /// \param source Target pointcloud of dtype Float32. It must contain
    /// normals.
    /// \param correspondences CorrespondenceSet: a pair of Int64 {C,}
    /// shape tensor.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &correspondences) const override;

    /// \brief Estimates the transformation matrix for PointToPoint method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud of dtype Float32.
    /// \param source Target pointcloud of dtype Float32.
    /// \param correspondences CorrespondenceSet: a pair of Int64 {C,} shape
    /// tensor.
    /// \return transformation between source to target, a tensor of
    /// shape {4, 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &correspondences) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// \class TransformationEstimationPointToPlane
///
/// Class to estimate a transformation of shape {4, 4} and dtype Float64 for
/// point to plane distance.
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
    /// \brief Computes RMSE (double) for PointToPlane method, between two
    /// pointclouds of type Float32, given CorrespondenceSet.
    ///
    /// \param source Source pointcloud of dtype Float32.
    /// \param source Target pointcloud of dtype Float32. It must contain
    /// normals.
    /// \param correspondences CorrespondenceSet: a pair of Int64 {C,}
    /// shape tensor.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &correspondences) const override;

    /// \brief Estimates the transformation matrix for PointToPlane method,
    /// a tensor of shape {4, 4}, and dtype Float64 on CPU device.
    ///
    /// \param source Source pointcloud of dtype Float32.
    /// \param source Target pointcloud of dtype Float32. It must contain
    /// normals.
    /// \param correspondences CorrespondenceSet: a pair of Int64 {C,}
    /// shape tensor.
    /// \return transformation between source to target, a tensor
    /// of shape {4, 4}, type Float64 on CPU device.
    core::Tensor ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &correspondences) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
