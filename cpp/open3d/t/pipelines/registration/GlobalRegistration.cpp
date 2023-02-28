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
        const RANSACConvergenceCriteria &criteria,
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration) {
    // TODO(wei): dimension check
    core::Tensor correspondences =
            CorrespondencesFromFeatures(source_feats, target_feats);
    return RANSACFromCorrespondences(source, target, correspondences,
                                     max_correspondence_distance, criteria,
                                     callback_after_iteration);
}

RegistrationResult RANSACFromCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const double max_correspondence_distance,
        const RANSACConvergenceCriteria &criteria,
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration) {
    const TransformationEstimation &estimation =
            TransformationEstimationPointToPoint();

    int n = correspondences.GetLength();

    // TODO: device check
    // core::Device device = source.GetDevice();

    // TODO: allow parameter (?)
    const int ransac_n = 3;

    core::Device host("CPU:0");

    // Move to host for easier memcopy/computation on small sized sample sets
    t::geometry::PointCloud source_host = source.To(host);
    t::geometry::PointCloud target_host = target.To(host);
    core::Tensor corres_host = correspondences.To(host);

    // Dummy correspondences to fit in EvaluateRegistration
    const core::Tensor dummy_corres =
            core::Tensor::Arange(0, ransac_n, 1, core::Int64, host);

    utility::random::UniformIntGenerator<int> rand_gen(0, n - 1);
    auto best_result = RegistrationResult();
    for (int itr = 0; itr < criteria.max_iteration_; ++itr) {
        std::vector<int64_t> source_indices_vec(ransac_n);
        std::vector<int64_t> target_indices_vec(ransac_n);

        // TODO(wei): random tensor generation in Tensor.h
        for (int s = 0; s < ransac_n; ++s) {
            int k = rand_gen();
            auto corres_k = corres_host[k];
            source_indices_vec[s] = corres_k[0].Item<int64_t>();
            target_indices_vec[s] = corres_k[1].Item<int64_t>();
        }

        core::Tensor source_indices(source_indices_vec, {ransac_n}, core::Int64,
                                    host);
        core::Tensor target_indices(target_indices_vec, {ransac_n}, core::Int64,
                                    host);

        t::geometry::PointCloud source_sample =
                source_host.SelectByIndex(source_indices);
        t::geometry::PointCloud target_sample =
                target_host.SelectByIndex(target_indices);

        // Inexpensive model estimation: on host
        core::Tensor transformation = estimation.ComputeTransformation(
                source_sample, target_sample, dummy_corres);

        // TODO: check for filtering
        // Inexpensive candidate check: on host

        // Expensive validation: on device
        auto result = EvaluateRegistration(
                source, target, max_correspondence_distance, transformation);

        // TODO: update validation
        if (result.IsBetterThan(best_result)) {
            best_result = result;
            utility::LogInfo(
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
