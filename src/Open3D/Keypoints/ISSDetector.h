// @file      ISSDetector.h
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#pragma once

#include <memory>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"

namespace open3d {
namespace keypoints {

/// \class ISSDetector
///
/// \brief ISS keypoint detector class, works in input point clouds. This
/// implements the keypoint detection modules proposed in Yu Zhong ,"Intrinsic
/// Shape Signatures: A Shape Descriptor for 3D Object Recognition", 2009.
/// The implementation is heavily inspred in the PCL implementation.
class ISSDetector {
public:
    /// \brief Parametrized Constructor. It creates a KDTree when it gets
    /// invoked, the KDTree will be used later on to compute the keypoints.
    ///
    /// \param cloud Input point cloud, agnostic to normal information.
    /// \param salient_radius The radius of the spherical
    /// neighborhood used to detect the keypoints \param non_max_radius
    /// The non maxima supression radius.
    /// If non of the input parameters are specified or are 0.0, then they will
    /// be computed from the input data, taking into account the Model
    /// Resolution.
    explicit ISSDetector(const std::shared_ptr<geometry::PointCloud>& cloud,
                         double salient_radius = 0.0,
                         double non_max_radius = 0.0)
        : cloud_(cloud),
          kdtree_(*cloud),
          salient_radius_(salient_radius),
          non_max_radius_(non_max_radius) {
        if (salient_radius_ == 0.0 || non_max_radius_ == 0.0) {
            const double resolution = ComputeModelResolution();
            salient_radius_ = 6 * resolution;
            non_max_radius_ = 4 * resolution;
        }
    }

    /// \brief Function to compute ISS keypoints for a point cloud.
    std::shared_ptr<geometry::PointCloud> ComputeKeypoints() const;

protected:
    /// \brief Helper function to compute the scatter matrix for a a point in
    /// the input pointcloud.
    ///
    /// \param p The 3D center point.
    Eigen::Matrix3d ComputeScatterMatrix(const Eigen::Vector3d& p) const;

    /// \brief Function to compute the model resolution;
    double ComputeModelResolution() const;

    /// Input PointCloud where to extract the keypoints
    std::shared_ptr<geometry::PointCloud> cloud_;

    /// KDTree to accelerate nearest neighbour searches
    geometry::KDTreeFlann kdtree_;

public:
    /// The radius of the spherical neighborhood used to detect keypoints.
    double salient_radius_ = 0.0;
    /// The non maxima suppression radius.
    double non_max_radius_ = 0.0;
    /// The upper bound on the ratio between the second and the first eigenvalue
    double gamma_21_ = 0.975;
    /// The upper bound on the ratio between the third and the second eigenvalue
    double gamma_32_ = 0.975;
    /// Minimum number of neighbors that has to be found to consider a keypoint.
    int min_neighbors_ = 5;
};

/// \brief Function that computes the ISS Keypoints from an input point cloud.
/// This implements the keypoint detection modules proposed in Yu Zhong
/// ,"Intrinsic Shape Signatures: A Shape Descriptor for 3D Object Recognition",
/// 2009. The implementation is heavily inspred in the PCL implementation.
//
/// \param input The input point cloud, agnostic to normal information.
/// \param salient_radius The radius of the spherical neighborhood used to
/// detect the keypoints \param non_max_radius The non maxima supression
/// radius.
/// If non of the input parameters are specified or are 0.0, then they will
/// be computed from the input data, taking into account the Model
/// Resolution.
std::shared_ptr<geometry::PointCloud> ComputeISSKeypoints(
        const std::shared_ptr<geometry::PointCloud>& input,
        double salient_radius = 0.0,
        double non_max_radius = 0.0);

}  // namespace keypoints
}  // namespace open3d
