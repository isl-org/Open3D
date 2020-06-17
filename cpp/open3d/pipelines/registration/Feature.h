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
#include <vector>

#include "Open3D/Geometry/KDTreeSearchParam.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

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
}  // namespace open3d
