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

#include "Open3D/Registration/CorrespondenceChecker.h"

#include <Eigen/Dense>

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace registration {

bool CorrespondenceCheckerBasedOnEdgeLength::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d & /*transformation*/) const {
    for (size_t i = 0; i < corres.size(); i++) {
        for (size_t j = i + 1; j < corres.size(); j++) {
            // check edge ij
            double dis_source = (source.points_[corres[i](0)] -
                                 source.points_[corres[j](0)])
                                        .norm();
            double dis_target = (target.points_[corres[i](1)] -
                                 target.points_[corres[j](1)])
                                        .norm();
            if (dis_source < dis_target * similarity_threshold_ ||
                dis_target < dis_source * similarity_threshold_) {
                return false;
            }
        }
    }
    return true;
}

bool CorrespondenceCheckerBasedOnDistance::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &transformation) const {
    for (const auto &c : corres) {
        const auto &pt = source.points_[c(0)];
        Eigen::Vector3d pt_trans =
                (transformation * Eigen::Vector4d(pt(0), pt(1), pt(2), 1.0))
                        .block<3, 1>(0, 0);
        if ((target.points_[c(1)] - pt_trans).norm() > distance_threshold_) {
            return false;
        }
    }
    return true;
}

bool CorrespondenceCheckerBasedOnNormal::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &transformation) const {
    if (source.HasNormals() == false || target.HasNormals() == false) {
        utility::LogWarning(
                "[CorrespondenceCheckerBasedOnNormal::Check] Pointcloud has no "
                "normals.");
        return true;
    }
    double cos_normal_angle_threshold = std::cos(normal_angle_threshold_);
    for (const auto &c : corres) {
        const auto &normal = source.normals_[c(0)];
        Eigen::Vector3d normal_trans =
                (transformation *
                 Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0))
                        .block<3, 1>(0, 0);
        if (target.normals_[c(1)].dot(normal_trans) <
            cos_normal_angle_threshold) {
            return false;
        }
    }
    return true;
}

}  // namespace registration
}  // namespace open3d
