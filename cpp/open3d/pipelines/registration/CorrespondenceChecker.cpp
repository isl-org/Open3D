// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/CorrespondenceChecker.h"

#include <Eigen/Dense>

#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
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
    if (!source.HasNormals() || !target.HasNormals()) {
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
}  // namespace pipelines
}  // namespace open3d
