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
#include "open3d/utility/Random.h"

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
    int n = correspondences.GetLength();

    // TODO: device check
    core::Device device = source.GetDevice();

    // TODO: allow parameter (?)
    const int ransac_n = 3;

    core::Tensor source_positions = source.GetPointPositions();
    core::Tensor target_positions = target.GetPointPositions();

    core::Tensor source_sample_positions({ransac_n, 3}, core::Dtype::Float32,
                                         device);
    core::Tensor target_sample_positions({ransac_n, 3}, core::Dtype::Float32,
                                         device);
    core::Tensor corres_sample =
            core::Tensor::Arange(0, ransac_n, 1, core::Int64, device);

    utility::random::UniformIntGenerator<int> rand_gen(0, n - 1);

    auto best_result = RegistrationResult();

    // On GPU: iteration without omp
    for (int itr = 0; itr < criteria.max_iteration_; ++itr) {
        // Construct point clouds to fit estimation.ComputeTransformation's
        // signature
        for (int s = 0; s < ransac_n; ++s) {
            int k = rand_gen();
            int i = correspondences[k][0].Item<int64_t>();
            int j = correspondences[k][1].Item<int64_t>();
            source_sample_positions.SetItem({core::TensorKey::Index(s)},
                                            source_positions[i]);
            target_sample_positions.SetItem({core::TensorKey::Index(s)},
                                            target_positions[j]);
        }

        t::geometry::PointCloud source_sample(source_sample_positions);
        t::geometry::PointCloud target_sample(target_sample_positions);

        core::Tensor transformation = estimation.ComputeTransformation(
                source_sample, target_sample, corres_sample);

        // TODO: check
        auto result = EvaluateRegistration(
                source, target, max_correspondence_distance, transformation);

        // TODO: update validation
        if (result.IsBetterThan(best_result)) {
            best_result = result;
            utility::LogDebug(
                    "RANSAC result updated at {:d} iters, current fitness = "
                    "{:e}, rmse = {:e}",
                    itr, result.fitness_, result.inlier_rmse_);
        }
    }
    return best_result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
