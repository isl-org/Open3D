// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "open3d/geometry/KDTreeSearchParam.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

typedef std::vector<Eigen::Vector2i> CorrespondenceSet;

/// \class Feature
///
/// \brief Class to store featrues for registration.
class Feature {
public:
    /// Resize feature data buffer to `dim x n`.
    ///
    /// \param dim Feature dimension per point.
    /// \param n Number of points.
    void Resize(int dim, int n) {
        data_.resize(dim, n);
        data_.setZero();
    }
    /// Returns feature dimensions per point.
    size_t Dimension() const { return data_.rows(); }
    /// Returns number of points.
    size_t Num() const { return data_.cols(); }

public:
    /// Data buffer storing features.
    Eigen::MatrixXd data_;
};

/// Function to compute FPFH feature for a point cloud.
///
/// \param input The Input point cloud.
/// \param search_param KDTree KNN search parameter.
std::shared_ptr<Feature> ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeSearchParam &search_param =
                geometry::KDTreeSearchParamKNN());

/// \brief Function to find correspondences via 1-nearest neighbor feature
/// matching. Target is used to construct a nearest neighbor search
/// object, in order to query source.
/// \param source_features (D, N) feature
/// \param target_features (D, M) feature
/// \param mutual_filter Boolean flag, only return correspondences s.t. i and j
/// are mutually the nearest neighbor
/// \return A CorrespondenceSet, where the first column is arange(0, N) of
/// source, and the second column is the corresponding index of target.
CorrespondenceSet CorrespondencesFromFeatures(const Feature &source_features,
                                              const Feature &target_features,
                                              bool mutual_filter = false);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
