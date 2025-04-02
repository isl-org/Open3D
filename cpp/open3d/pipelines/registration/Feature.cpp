// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/Feature.h"

#include <Eigen/Dense>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace pipelines {
namespace registration {

std::shared_ptr<Feature> Feature::SelectByIndex(
        const std::vector<size_t> &indices, bool invert /* = false */) const {
    auto output = std::make_shared<Feature>();

    std::vector<bool> mask = std::vector<bool>(data_.cols(), invert);
    size_t n_output_features = 0;
    for (size_t i : indices) {
        if (i < mask.size()) {
            if (mask[i] == invert) {
                mask[i] = !invert;
                n_output_features++;
            }
        } else {
            utility::LogWarning(
                    "[SelectByIndex] contains index {} that is "
                    "not within the bounds",
                    (int)i);
        }
    }
    if (invert) {
        n_output_features = data_.cols() - n_output_features;
    }

    output->Resize(data_.rows(), n_output_features);

    for (size_t i = 0, current_col_feature = 0;
         i < static_cast<size_t>(data_.cols()); i++) {
        if (mask[i]) {
            output->data_.col(current_col_feature++) = data_.col(i);
        }
    }

    utility::LogDebug(
            "[SelectByIndex] Feature group down sampled from {:d} features to "
            "{:d} features.",
            (int)data_.cols(), (int)output->data_.cols());

    return output;
}

static Eigen::Vector4d ComputePairFeatures(const Eigen::Vector3d &p1,
                                           const Eigen::Vector3d &n1,
                                           const Eigen::Vector3d &p2,
                                           const Eigen::Vector3d &n2) {
    Eigen::Vector4d result;
    Eigen::Vector3d dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    auto n1_copy = n1;
    auto n2_copy = n2;
    double angle1 = n1_copy.dot(dp2p1) / result(3);
    double angle2 = n2_copy.dot(dp2p1) / result(3);
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    double v_norm = v.norm();
    if (v_norm == 0.0) {
        return Eigen::Vector4d::Zero();
    }
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

static std::shared_ptr<Feature> ComputeSPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeFlann &kdtree,
        const geometry::KDTreeSearchParam &search_param,
        const utility::optional<std::vector<size_t>> &indices =
                utility::nullopt) {
    const bool filter_spfh = indices.has_value();
    const auto spfh_indices = indices.value_or(std::vector<size_t>());

    const size_t n_spfh =
            filter_spfh ? spfh_indices.size() : input.points_.size();
    auto feature = std::make_shared<Feature>();
    feature->Resize(33, (int)n_spfh);

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)n_spfh; i++) {
        const int point_idx = filter_spfh ? spfh_indices[i] : i;
        const auto &point = input.points_[point_idx];
        const auto &normal = input.normals_[point_idx];
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(point, search_param, indices, distance2) > 1) {
            // only compute SPFH feature when a point has neighbors
            double hist_incr = 100.0 / (double)(indices.size() - 1);
            for (size_t k = 1; k < indices.size(); k++) {
                // skip the point itself, compute histogram
                auto pf = ComputePairFeatures(point, normal,
                                              input.points_[indices[k]],
                                              input.normals_[indices[k]]);
                int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 11, i) += hist_incr;
                h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
                if (h_index < 0) h_index = 0;
                if (h_index >= 11) h_index = 10;
                feature->data_(h_index + 22, i) += hist_incr;
            }
        }
    }
    return feature;
}

std::shared_ptr<Feature> ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeSearchParam
                &search_param /* = geometry::KDTreeSearchParamKNN()*/,
        const utility::optional<std::vector<size_t>>
                &indices /* = utility::nullopt*/) {
    if (!input.HasNormals()) {
        utility::LogError("Failed because input point cloud has no normal.");
    }

    const bool filter_fpfh = indices.has_value();
    std::vector<int> fpfh_indices;
    if (filter_fpfh) {
        std::vector<bool> mask_fpfh(input.points_.size(), false);
        for (auto idx : indices.value()) {
            if (idx < mask_fpfh.size()) {
                if (!mask_fpfh[idx]) {
                    mask_fpfh[idx] = true;
                }
            } else {
                utility::LogWarning(
                        "[ComputeFPFHFeature] contains index {} that is "
                        "not within the bounds",
                        idx);
            }
        }
        fpfh_indices.reserve(indices.value().size());
        for (size_t i = 0; i < mask_fpfh.size(); i++) {
            if (mask_fpfh[i]) {
                fpfh_indices.push_back(i);
            }
        }
    }

    const size_t n_fpfh =
            filter_fpfh ? fpfh_indices.size() : input.points_.size();

    geometry::KDTreeFlann kdtree(input);

    std::vector<size_t> spfh_indices;
    std::vector<int> map_point_idx_to_spfh_idx;
    std::vector<std::vector<int>> map_fpfh_idx_to_indices;
    std::vector<std::vector<double>> map_fpfh_idx_to_distance2;
    if (filter_fpfh) {
        // compute neighbors of the selected points
        // using vector<uint8_t> as a boolean mask for the parallel loop
        // since vector<bool> is not thread safe in writing.
        std::vector<uint8_t> mask_spfh(input.points_.size(), 0);
        map_fpfh_idx_to_indices = std::vector<std::vector<int>>(n_fpfh);
        map_fpfh_idx_to_distance2 = std::vector<std::vector<double>>(n_fpfh);
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int i = 0; i < (int)n_fpfh; i++) {
            const auto &point = input.points_[fpfh_indices[i]];
            std::vector<int> p_indices;
            std::vector<double> p_distance2;
            kdtree.Search(point, search_param, p_indices, p_distance2);
            for (size_t k = 0; k < p_indices.size(); k++) {
                if (!mask_spfh[p_indices[k]]) {
                    mask_spfh[p_indices[k]] = 1;
                }
            }
            map_fpfh_idx_to_indices[i] = std::move(p_indices);
            map_fpfh_idx_to_distance2[i] = std::move(p_distance2);
        }
        size_t spfh_indices_reserve_factor;
        switch (search_param.GetSearchType()) {
            case geometry::KDTreeSearchParam::SearchType::Knn:
                spfh_indices_reserve_factor =
                        ((const geometry::KDTreeSearchParamKNN &)search_param)
                                .knn_;
                break;
            case geometry::KDTreeSearchParam::SearchType::Hybrid:
                spfh_indices_reserve_factor =
                        ((const geometry::KDTreeSearchParamHybrid &)
                                 search_param)
                                .max_nn_;
                break;
            default:
                spfh_indices_reserve_factor = 30;
        }
        spfh_indices.reserve(spfh_indices_reserve_factor * fpfh_indices.size());
        map_point_idx_to_spfh_idx = std::vector<int>(input.points_.size(), -1);
        for (size_t i = 0; i < mask_spfh.size(); i++) {
            if (mask_spfh[i]) {
                map_point_idx_to_spfh_idx[i] = spfh_indices.size();
                spfh_indices.push_back(i);
            }
        }
    }

    auto feature = std::make_shared<Feature>();
    feature->Resize(33, (int)n_fpfh);

    auto spfh = filter_fpfh ? ComputeSPFHFeature(input, kdtree, search_param,
                                                 spfh_indices)
                            : ComputeSPFHFeature(input, kdtree, search_param);
    if (spfh == nullptr) {
        utility::LogError("Internal error: SPFH feature is nullptr.");
    }
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)n_fpfh; i++) {
        int i_spfh;
        std::vector<int> p_indices;
        std::vector<double> p_distance2;
        if (filter_fpfh) {
            i_spfh = map_point_idx_to_spfh_idx[fpfh_indices[i]];
            p_indices = std::move(map_fpfh_idx_to_indices[i]);
            p_distance2 = std::move(map_fpfh_idx_to_distance2[i]);
        } else {
            i_spfh = i;
            kdtree.Search(input.points_[i], search_param, p_indices,
                          p_distance2);
        }
        if (p_indices.size() > 1) {
            double sum[3] = {0.0, 0.0, 0.0};
            for (size_t k = 1; k < p_indices.size(); k++) {
                // skip the point itself
                double dist = p_distance2[k];
                if (dist == 0.0) continue;
                int p_index_k =
                        filter_fpfh ? map_point_idx_to_spfh_idx[p_indices[k]]
                                    : p_indices[k];
                for (int j = 0; j < 33; j++) {
                    double val = spfh->data_(j, p_index_k) / dist;
                    sum[j / 11] += val;
                    feature->data_(j, i) += val;
                }
            }
            for (int j = 0; j < 3; j++)
                if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
            for (int j = 0; j < 33; j++) {
                feature->data_(j, i) *= sum[j / 11];
                // The commented line is the fpfh function in the paper.
                // But according to PCL implementation, it is skipped.
                // Our initial test shows that the full fpfh function in the
                // paper seems to be better than PCL implementation. Further
                // test required.
                feature->data_(j, i) += spfh->data_(j, i_spfh);
            }
        }
    }

    utility::LogDebug(
            "[ComputeFPFHFeature] Computed {:d} features from "
            "input point cloud with {:d} points.",
            (int)feature->data_.cols(), (int)input.points_.size());

    return feature;
}

CorrespondenceSet CorrespondencesFromFeatures(const Feature &source_features,
                                              const Feature &target_features,
                                              bool mutual_filter,
                                              float mutual_consistent_ratio) {
    const int num_searches = mutual_filter ? 2 : 1;

    // Access by reference, since Eigen Matrix could be copied
    std::array<std::reference_wrapper<const Feature>, 2> features{
            std::reference_wrapper<const Feature>(source_features),
            std::reference_wrapper<const Feature>(target_features)};
    std::array<int, 2> num_pts{int(source_features.data_.cols()),
                               int(target_features.data_.cols())};
    std::vector<CorrespondenceSet> corres(num_searches);

    const int kMaxThreads = utility::EstimateMaxThreads();
    const int kOuterThreads = std::min(kMaxThreads, num_searches);
    const int kInnerThreads = std::max(kMaxThreads / num_searches, 1);
    (void)kOuterThreads;  // Avoids compiler warning if OpenMP is disabled
    (void)kInnerThreads;
#pragma omp parallel for num_threads(kOuterThreads)
    for (int k = 0; k < num_searches; ++k) {
        geometry::KDTreeFlann kdtree(features[1 - k]);

        int num_pts_k = num_pts[k];
        corres[k] = CorrespondenceSet(num_pts_k);
#pragma omp parallel for num_threads(kInnerThreads)
        for (int i = 0; i < num_pts_k; i++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            kdtree.SearchKNN(Eigen::VectorXd(features[k].get().data_.col(i)), 1,
                             corres_tmp, dist_tmp);
            int j = corres_tmp[0];
            corres[k][i] = Eigen::Vector2i(i, j);
        }
    }

    // corres[0]: corres_ij, corres[1]: corres_ji
    if (!mutual_filter) return corres[0];

    // should not use parallel for due to emplace back
    CorrespondenceSet corres_mutual;
    int num_src_pts = num_pts[0];
    for (int i = 0; i < num_src_pts; ++i) {
        int j = corres[0][i](1);
        if (corres[1][j](1) == i) {
            corres_mutual.emplace_back(i, j);
        }
    }

    // Empirically mutual correspondence set should not be too small
    if (int(corres_mutual.size()) >=
        int(mutual_consistent_ratio * num_src_pts)) {
        utility::LogDebug("{:d} correspondences remain after mutual filter",
                          corres_mutual.size());
        return corres_mutual;
    }
    utility::LogWarning(
            "Too few correspondences ({:d}) after mutual filter, fall back to "
            "original correspondences.",
            corres_mutual.size());
    return corres[0];
}
}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
