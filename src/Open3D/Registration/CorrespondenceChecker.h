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
#include <vector>

#include "Open3D/Registration/TransformationEstimation.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace registration {

/// \class CorrespondenceChecker
///
/// \brief Base class that checks if two (small) point clouds can be aligned.
///
/// This class is used in feature based matching algorithms (such as RANSAC and
/// FastGlobalRegistration) to prune out outlier correspondences.
/// The virtual function Check() must be implemented in subclasses.
class CorrespondenceChecker {
public:
    /// \brief Default Constructor.
    ///
    /// \param require_pointcloud_alignment Specifies whether point cloud
    /// alignment is required.
    CorrespondenceChecker(bool require_pointcloud_alignment)
        : require_pointcloud_alignment_(require_pointcloud_alignment) {}
    virtual ~CorrespondenceChecker() {}

public:
    /// \brief Function to check if two points can be aligned.
    ///
    /// The two input point
    /// clouds must have exact the same number of points.
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param corres Correspondence set between source and target point cloud.
    /// \param transformation The estimated transformation (inplace).
    virtual bool Check(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres,
                       const Eigen::Matrix4d &transformation) const = 0;

public:
    /// Some checkers do not require point clouds to be aligned, e.g., the edge
    /// length checker. Some checkers do, e.g., the distance checker.
    bool require_pointcloud_alignment_;
};

/// \class CorrespondenceCheckerBasedOnEdgeLength
///
/// \brief Check if two point clouds build the polygons with similar edge
/// lengths.
///
/// That is, checks if the lengths of any two arbitrary edges (line formed by
/// two vertices) individually drawn withinin source point cloud and within the
/// target point cloud with correspondences are similar. The only parameter
/// similarity_threshold is a number between 0 (loose) and 1 (strict).
class CorrespondenceCheckerBasedOnEdgeLength : public CorrespondenceChecker {
public:
    /// \brief Default Constructor.
    ///
    /// \param similarity_threshold specifies the threshold within which 2
    /// arbitrary edges are similar.
    CorrespondenceCheckerBasedOnEdgeLength(double similarity_threshold = 0.9)
        : CorrespondenceChecker(false),
          similarity_threshold_(similarity_threshold) {}
    ~CorrespondenceCheckerBasedOnEdgeLength() override {}

public:
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const CorrespondenceSet &corres,
               const Eigen::Matrix4d &transformation) const override;

public:
    /// For the check to be true,
    /// ||edgesource||>similarity_threshold×||edgetarget|| and
    /// ||edgetarget||>similarity_threshold×||edgesource|| must hold true for
    /// all edges.
    double similarity_threshold_;
};

/// \class CorrespondenceCheckerBasedOnDistance
///
/// \brief Check if two aligned point clouds are close.
class CorrespondenceCheckerBasedOnDistance : public CorrespondenceChecker {
public:
    /// \brief Default Constructor.
    ///
    /// \param distance_threshold Distance threashold for the check.
    CorrespondenceCheckerBasedOnDistance(double distance_threshold)
        : CorrespondenceChecker(true),
          distance_threshold_(distance_threshold) {}
    ~CorrespondenceCheckerBasedOnDistance() override {}

public:
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const CorrespondenceSet &corres,
               const Eigen::Matrix4d &transformation) const override;

public:
    /// Distance threashold for the check.
    double distance_threshold_;
};

/// \class CorrespondenceCheckerBasedOnNormal
///
/// \brief Class to check if two aligned point clouds have similar normals.
///
/// It considers vertex normal affinity of any correspondences. It computes dot
/// product of two normal vectors. It takes radian value for the threshold.
class CorrespondenceCheckerBasedOnNormal : public CorrespondenceChecker {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param normal_angle_threshold Radian value for angle threshold.
    CorrespondenceCheckerBasedOnNormal(double normal_angle_threshold)
        : CorrespondenceChecker(true),
          normal_angle_threshold_(normal_angle_threshold) {}
    ~CorrespondenceCheckerBasedOnNormal() override {}

public:
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const CorrespondenceSet &corres,
               const Eigen::Matrix4d &transformation) const override;

public:
    /// Radian value for angle threshold.
    double normal_angle_threshold_;
};

}  // namespace registration
}  // namespace open3d
