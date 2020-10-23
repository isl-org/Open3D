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
#include <vector>

#include "open3d/geometry/Image.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/color_map/EigenHelperForNonRigidOptimization.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace pipelines {
namespace color_map {

class ImageWarpingField;

class ColorMapOptimizationJacobian {
public:
    ColorMapOptimizationJacobian() {}

public:
    /// Function to compute i-th row of J and r
    /// the vector form of J_r is basically 6x1 matrix, but it can be
    /// easily extendable to 6xn matrix.
    /// See RGBDOdometryJacobianFromHybridTerm for this case.
    void ComputeJacobianAndResidualRigid(
            int row,
            Eigen::Vector6d& J_r,
            double& r,
            double& w,
            const geometry::TriangleMesh& mesh,
            const std::vector<double>& proxy_intensity,
            const std::shared_ptr<geometry::Image>& images_gray,
            const std::shared_ptr<geometry::Image>& images_dx,
            const std::shared_ptr<geometry::Image>& images_dy,
            const Eigen::Matrix4d& intrinsic,
            const Eigen::Matrix4d& extrinsic,
            const std::vector<int>& visibility_image_to_vertex,
            const int image_boundary_margin);

    /// Function to compute i-th row of J and r
    /// The vector form of J_r is basically 14x1 matrix.
    /// This function can take additional matrix multiplication pattern
    /// to avoid full matrix multiplication
    void ComputeJacobianAndResidualNonRigid(
            int row,
            Eigen::Vector14d& J_r,
            double& r,
            Eigen::Vector14i& pattern,
            const geometry::TriangleMesh& mesh,
            const std::vector<double>& proxy_intensity,
            const std::shared_ptr<geometry::Image>& images_gray,
            const std::shared_ptr<geometry::Image>& images_dx,
            const std::shared_ptr<geometry::Image>& images_dy,
            const ImageWarpingField& warping_fields,
            const ImageWarpingField& warping_fields_init,
            const Eigen::Matrix4d& intrinsic,
            const Eigen::Matrix4d& extrinsic,
            const std::vector<int>& visibility_image_to_vertex,
            const int image_boundary_margin);
};

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
