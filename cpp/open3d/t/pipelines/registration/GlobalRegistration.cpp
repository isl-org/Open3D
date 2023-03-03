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
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/utility/Random.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace registration {

bool ConsistencyCheck(const int ransac_n,
                      const t::geometry::PointCloud &source_samples,
                      const t::geometry::PointCloud &target_samples,
                      const core::Tensor &transformation,
                      const float dist_threshold,
                      // const float cos_normal_angle_threshold,
                      const float edge_similarity_ratio = 0.9) {
    auto source_trans_samples = source_samples.Clone();
    source_trans_samples.Transform(transformation);

    // Element-wise constraints
    core::Tensor source_trans_positions =
            source_trans_samples.GetPointPositions();
    core::Tensor target_positions = target_samples.GetPointPositions();

    auto position_diff = (source_trans_positions - target_positions);
    auto squared_dist = (position_diff * position_diff).Sum({1});
    float squared_dist_threshold = dist_threshold * dist_threshold;
    // utility::LogInfo("squared_dist: {}, thr: {}", squared_dist.ToString(),
    //                  squared_dist_threshold);

    if ((squared_dist.Gt(squared_dist_threshold).Any()).Item<bool>())
        return false;

    // core::Tensor source_trans_normals =
    // source_trans_samples.GetPointNormals(); core::Tensor target_normals =
    // target_samples.GetPointNormals();

    // auto normal_dot = (source_trans_normals * target_normals).Sum({1});
    // if (normal_dot.Max() < cos_normal_angle_threshold) return false;

    // Edge constraints
    std::vector<int64_t> edge_i_vec, edge_j_vec;
    for (int i = 0; i < ransac_n; ++i) {
        for (int j = i + 1; j < ransac_n; ++j) {
            edge_i_vec.push_back(i);
            edge_j_vec.push_back(j);
        }
    }

    // Vectorize
    core::Tensor edge_i_indices(
            edge_i_vec, {static_cast<int64_t>(edge_i_vec.size())}, core::Int64);
    core::Tensor edge_j_indices(
            edge_j_vec, {static_cast<int64_t>(edge_j_vec.size())}, core::Int64);

    auto source_trans_i = source_trans_positions.IndexGet({edge_i_indices});
    auto source_trans_j = source_trans_positions.IndexGet({edge_j_indices});
    auto diff_source_ij = source_trans_i - source_trans_j;
    // utility::LogInfo("diff source_ij shape = {}", diff_source_ij.GetShape());

    auto dist_source_ij = (diff_source_ij * diff_source_ij).Sum({1});

    auto target_i = target_positions.IndexGet({edge_i_indices});
    auto target_j = target_positions.IndexGet({edge_j_indices});
    auto diff_target_ij = target_i - target_j;
    auto dist_target_ij = (diff_target_ij * diff_target_ij).Sum({1});

    float squared_similarity_ratio =
            edge_similarity_ratio * edge_similarity_ratio;
    auto inconsistency_st =
            dist_source_ij < dist_target_ij * squared_similarity_ratio;
    auto inconsistency_ts =
            dist_target_ij < dist_source_ij * squared_similarity_ratio;
    // utility::LogInfo("edge consistency src2dst: {}, dst2src: {}",
    //                  dist_source_ij.ToString(), dist_target_ij.ToString());
    if (inconsistency_st.LogicalOr(inconsistency_ts).Any().Item<bool>())
        return false;

    return true;
}

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

    utility::LogInfo("source device: {}", source.GetDevice().ToString());
    utility::LogInfo("target device: {}", target.GetDevice().ToString());

    core::Device host("CPU:0");

    // Move to host for easier memcopy/computation
    // on small sized sample sets
    t::geometry::PointCloud source_host = source.To(host);
    t::geometry::PointCloud target_host = target.To(host);
    core::Tensor corres_host = correspondences.To(host);

    // Dummy correspondences to fit in
    // EvaluateRegistration
    const core::Tensor dummy_corres =
            core::Tensor::Arange(0, ransac_n, 1, core::Int64, host);

    RegistrationResult best_result;
    core::nns::NearestNeighborSearch target_nns(target.GetPointPositions());
    target_nns.HybridIndex(max_correspondence_distance);

    int total_validations = 0;
    float total_proposal_time = 0;
    float total_transform_time = 0;
    float total_check_time = 0;
    float total_validation_time = 0;
#pragma omp parallel
    {
        RegistrationResult best_local_result;
        utility::random::UniformIntGenerator<int> rand_gen(0, n - 1);
        int local_validations = 0;

        float local_proposal_time = 0;
        float local_transform_time = 0;
        float local_check_time = 0;
        float local_validation_time = 0;
        utility::Timer timer;

#pragma omp for nowait
        for (int itr = 0; itr < criteria.max_iteration_; ++itr) {
            std::vector<int64_t> source_indices_vec(ransac_n);
            std::vector<int64_t> target_indices_vec(ransac_n);

            // TODO(wei): random tensor generation in
            // Tensor.h
            timer.Start();
            for (int s = 0; s < ransac_n; ++s) {
                int k = rand_gen();
                auto corres_k = corres_host[k];
                source_indices_vec[s] = corres_k[0].Item<int64_t>();
                target_indices_vec[s] = corres_k[1].Item<int64_t>();
            }

            core::Tensor source_indices(source_indices_vec, {ransac_n},
                                        core::Int64, host);
            core::Tensor target_indices(target_indices_vec, {ransac_n},
                                        core::Int64, host);

            t::geometry::PointCloud source_sample =
                    source_host.SelectByIndex(source_indices);
            t::geometry::PointCloud target_sample =
                    target_host.SelectByIndex(target_indices);
            timer.Stop();
            local_proposal_time += timer.GetDurationInSecond();

            // Inexpensive model estimation: on host
            timer.Start();
            core::Tensor transformation = estimation.ComputeTransformation(
                    source_sample, target_sample, dummy_corres);
            timer.Stop();
            local_transform_time += timer.GetDurationInSecond();

            // TODO: check for filtering
            // Inexpensive candidate check: on host
            timer.Start();
            bool consistent = ConsistencyCheck(
                    ransac_n, source_sample, target_sample, transformation,
                    max_correspondence_distance, 0.9);
            timer.Stop();
            local_check_time += timer.GetDurationInSecond();
            if (!consistent) continue;

            // Expensive validation: should be on device
            timer.Start();
            auto result = ComputeRegistrationResult(
                    source.Clone().Transform(transformation), target_nns,
                    max_correspondence_distance, transformation);
            timer.Stop();
            local_validations++;
            local_validation_time += timer.GetDurationInSecond();

            // TODO: update validation
            if (result.IsBetterThan(best_local_result)) {
                best_local_result = result;
                utility::LogInfo(
                        "RANSAC result updated at {:d} "
                        "iters, current fitness = "
                        "{:e}, rmse = {:e}",
                        itr, result.fitness_, result.inlier_rmse_);
            }
        }  // omp nowait

        // Reduce across threads
#pragma omp critical(RANSACFromCorrespondences)
        {
            if (best_local_result.IsBetterThan(best_result)) {
                best_result = best_local_result;
            }
            total_validations += local_validations;

            total_transform_time += local_transform_time;
            total_proposal_time += local_proposal_time;
            total_check_time += local_check_time;
            total_validation_time += local_validation_time;
        }
    }  // omp parallel

    utility::LogInfo("Total validations: {}", total_validations);
    utility::LogInfo(
            "Average proposal time: {}, transform time: {} check time: {}, "
            "validation time: {}",
            total_proposal_time / criteria.max_iteration_,
            total_transform_time / criteria.max_iteration_,
            total_check_time / criteria.max_iteration_,
            total_validation_time / total_validations);
    return best_result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
