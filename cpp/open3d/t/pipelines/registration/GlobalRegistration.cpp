// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/GlobalRegistration.h"

#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/registration/Feature.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace registration {

RegistrationResult RANSACFromFeatures(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &source_feats,
        const core::Tensor &target_feats,
        const double max_correspondence_distance,
        const TransformationEstimation &estimation,
        const RANSACConvergenceCriteria &criteria,
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration) {
    // TODO(wei): dimension check
    core::Tensor correspondences =
            CorrespondencesFromFeatures(source_feats, target_feats);
    return RANSACFromCorrespondences(source, target, correspondences,
                                     max_correspondence_distance, estimation,
                                     criteria, callback_after_iteration);
}

RegistrationResult RANSACFromCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const double max_correspondence_distance,
        const TransformationEstimation &estimation,
        const RANSACConvergenceCriteria &criteria,
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration) {
    // TODO(wei): implement
    return RegistrationResult();
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
