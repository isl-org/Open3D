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

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
