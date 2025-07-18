// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include "open3d/core/ParallelFor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/Feature.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace registration {

core::Tensor ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const utility::optional<int> max_nn,
        const utility::optional<double> radius,
        const utility::optional<core::Tensor> &indices) {
    core::AssertTensorDtypes(input.GetPointPositions(),
                             {core::Float64, core::Float32});
    if (max_nn.has_value() && max_nn.value() <= 3) {
        utility::LogError("max_nn must be greater than 3.");
    }
    if (radius.has_value() && radius.value() <= 0) {
        utility::LogError("radius must be greater than 0.");
    }
    if (!input.HasPointNormals()) {
        utility::LogError("The input point cloud has no normal.");
    }

    const int64_t num_points = input.GetPointPositions().GetLength();
    const core::Dtype dtype = input.GetPointPositions().GetDtype();
    const core::Device device = input.GetPointPositions().GetDevice();

    core::nns::NearestNeighborSearch tree(input.GetPointPositions(),
                                          core::Int32);
    bool tree_set = false;

    const bool filter_fpfh = indices.has_value();
    // If we are computing a subset of the FPFH feature,
    // cache some information to speed up the computation
    // if the ratio of the indices to the total number of points is high.
    const double cache_info_indices_ratio_thresh = 0.01;
    bool cache_fpfh_info = true;

    core::Tensor mask_fpfh_points;
    core::Tensor indices_fpfh_points;
    core::Tensor map_point_idx_to_required_point_idx;
    core::Tensor map_required_point_idx_to_point_idx;
    core::Tensor save_p_indices, save_p_distance2, save_p_counts;
    core::Tensor mask_spfh_points;

    // If we are computing a subset of the FPFH feature, we need to find
    // the subset of points (neighbors) required to compute the FPFH features.
    if (filter_fpfh) {
        if (indices.value().GetLength() == 0) {
            return core::Tensor::Zeros({0, 33}, dtype, device);
        }
        mask_fpfh_points =
                core::Tensor::Zeros({num_points}, core::Bool, device);
        mask_fpfh_points.IndexSet({indices.value()},
                                  core::Tensor::Ones({1}, core::Bool, device));
        const core::Tensor query_point_positions =
                input.GetPointPositions().IndexGet({mask_fpfh_points});
        core::Tensor p_indices, p_distance2, p_counts;
        if (radius.has_value() && max_nn.has_value()) {
            tree_set = tree.HybridIndex(radius.value());
            if (!tree_set) {
                utility::LogError("Building HybridIndex failed.");
            }
            std::tie(p_indices, p_distance2, p_counts) = tree.HybridSearch(
                    query_point_positions, radius.value(), max_nn.value());
        } else if (!radius.has_value() && max_nn.has_value()) {
            tree_set = tree.KnnIndex();
            if (!tree_set) {
                utility::LogError("Building KnnIndex failed.");
            }
            std::tie(p_indices, p_distance2) =
                    tree.KnnSearch(query_point_positions, max_nn.value());

            // Make counts full with min(max_nn, num_points).
            const int fill_value =
                    max_nn.value() > num_points ? num_points : max_nn.value();
            p_counts = core::Tensor::Full({query_point_positions.GetLength()},
                                          fill_value, core::Int32, device);
        } else if (radius.has_value() && !max_nn.has_value()) {
            tree_set = tree.FixedRadiusIndex(radius.value());
            if (!tree_set) {
                utility::LogError("Building RadiusIndex failed.");
            }
            std::tie(p_indices, p_distance2, p_counts) = tree.FixedRadiusSearch(
                    query_point_positions, radius.value());
        } else {
            utility::LogError("Both max_nn and radius are none.");
        }

        core::Tensor mask_required_points =
                core::Tensor::Zeros({num_points}, core::Bool, device);
        mask_required_points.IndexSet(
                {p_indices.To(core::Int64).View({-1})},
                core::Tensor::Ones({1}, core::Bool, device));
        map_required_point_idx_to_point_idx =
                mask_required_points.NonZero().GetItem(
                        {core::TensorKey::Index(0)});
        indices_fpfh_points =
                mask_fpfh_points.NonZero().GetItem({core::TensorKey::Index(0)});

        const bool is_radius_search = p_indices.GetShape().size() == 1;

        // Cache the info if the ratio of the indices to the total number of
        // points is high and we are not doing a radius search. Radius search
        // requires a different pipeline since tensor output p_counts is a
        // prefix sum.
        cache_fpfh_info = !is_radius_search &&
                          (indices_fpfh_points.GetLength() >=
                           cache_info_indices_ratio_thresh * num_points);

        if (cache_fpfh_info) {
            map_point_idx_to_required_point_idx =
                    core::Tensor::Full({num_points}, -1, core::Int32, device);
            map_point_idx_to_required_point_idx.IndexSet(
                    {map_required_point_idx_to_point_idx},
                    core::Tensor::Arange(
                            0, map_required_point_idx_to_point_idx.GetLength(),
                            1, core::Int32, device));

            core::SizeVector save_p_indices_shape = p_indices.GetShape();
            save_p_indices_shape[0] =
                    map_required_point_idx_to_point_idx.GetLength();
            save_p_indices = core::Tensor::Zeros(save_p_indices_shape,
                                                 core::Int32, device);
            save_p_distance2 = core::Tensor::Zeros(save_p_indices.GetShape(),
                                                   dtype, device);
            save_p_counts = core::Tensor::Zeros(
                    {map_required_point_idx_to_point_idx.GetLength() +
                     (is_radius_search ? 1 : 0)},
                    core::Int32, device);

            core::Tensor map_fpfh_point_idx_to_required_point_idx =
                    map_point_idx_to_required_point_idx
                            .IndexGet({indices_fpfh_points})
                            .To(core::Int64);

            save_p_indices.IndexSet({map_fpfh_point_idx_to_required_point_idx},
                                    p_indices);
            save_p_distance2.IndexSet(
                    {map_fpfh_point_idx_to_required_point_idx}, p_distance2);
            save_p_counts.IndexSet({map_fpfh_point_idx_to_required_point_idx},
                                   p_counts);

            // If we are filtering FPFH features, we have already computed some
            // info about the FPFH points' neighbors. Now we just need to
            // compute the info for the remaining required points, so skip the
            // computation for the already computed info.
            mask_spfh_points =
                    core::Tensor::Zeros({num_points}, core::Bool, device);
            mask_spfh_points.IndexSet(
                    {map_required_point_idx_to_point_idx},
                    core::Tensor::Ones({1}, core::Bool, device));
            mask_spfh_points.IndexSet(
                    {indices_fpfh_points},
                    core::Tensor::Zeros({1}, core::Bool, device));
        } else {
            mask_spfh_points = mask_required_points;
        }
    }

    const core::Tensor query_point_positions =
            filter_fpfh ? input.GetPointPositions().IndexGet({mask_spfh_points})
                        : input.GetPointPositions();

    // Compute nearest neighbors and squared distances.
    core::Tensor p_indices, p_distance2, p_counts;

    if (radius.has_value() && max_nn.has_value()) {
        if (!tree_set) {
            tree_set = tree.HybridIndex(radius.value());
            if (!tree_set) {
                utility::LogError("Building HybridIndex failed.");
            }
        }
        std::tie(p_indices, p_distance2, p_counts) = tree.HybridSearch(
                query_point_positions, radius.value(), max_nn.value());
        utility::LogDebug(
                "Use HybridSearch [max_nn: {} | radius {}] for computing FPFH "
                "feature.",
                max_nn.value(), radius.value());
    } else if (!radius.has_value() && max_nn.has_value()) {
        if (!tree_set) {
            tree_set = tree.KnnIndex();
            if (!tree_set) {
                utility::LogError("Building KnnIndex failed.");
            }
        }

        // tree.KnnSearch complains if the query point cloud is empty.
        if (query_point_positions.GetLength() > 0) {
            std::tie(p_indices, p_distance2) =
                    tree.KnnSearch(query_point_positions, max_nn.value());

            const int fill_value =
                    max_nn.value() > num_points ? num_points : max_nn.value();

            p_counts = core::Tensor::Full({query_point_positions.GetLength()},
                                          fill_value, core::Int32, device);
        } else {
            p_indices = core::Tensor::Zeros({0, max_nn.value()}, core::Int32,
                                            device);
            p_distance2 =
                    core::Tensor::Zeros({0, max_nn.value()}, dtype, device);
            p_counts = core::Tensor::Zeros({0}, core::Int32, device);
        }

        utility::LogDebug(
                "Use KNNSearch  [max_nn: {}] for computing FPFH feature.",
                max_nn.value());
    } else if (radius.has_value() && !max_nn.has_value()) {
        if (!tree_set) {
            tree_set = tree.FixedRadiusIndex(radius.value());
            if (!tree_set) {
                utility::LogError("Building RadiusIndex failed.");
            }
        }
        std::tie(p_indices, p_distance2, p_counts) =
                tree.FixedRadiusSearch(query_point_positions, radius.value());
        utility::LogDebug(
                "Use RadiusSearch [radius: {}] for computing FPFH feature.",
                radius.value());
    } else {
        utility::LogError("Both max_nn and radius are none.");
    }

    core::Tensor fpfh;
    if (filter_fpfh) {
        const int64_t size = indices_fpfh_points.GetLength();
        fpfh = core::Tensor::Zeros({size, 33}, dtype, device);
        core::Tensor final_p_indices, final_p_distance2, final_p_counts;
        if (cache_fpfh_info) {
            core::Tensor map_spfh_idx_to_required_point_idx =
                    map_point_idx_to_required_point_idx
                            .IndexGet({mask_spfh_points})
                            .To(core::Int64);
            save_p_indices.IndexSet({map_spfh_idx_to_required_point_idx},
                                    p_indices);
            save_p_distance2.IndexSet({map_spfh_idx_to_required_point_idx},
                                      p_distance2);
            save_p_counts.IndexSet({map_spfh_idx_to_required_point_idx},
                                   p_counts);
            final_p_indices = save_p_indices;
            final_p_distance2 = save_p_distance2;
            final_p_counts = save_p_counts;
        } else {
            final_p_indices = p_indices;
            final_p_distance2 = p_distance2;
            final_p_counts = p_counts;
        }
        pipelines::kernel::ComputeFPFHFeature(
                input.GetPointPositions(), input.GetPointNormals(),
                final_p_indices, final_p_distance2, final_p_counts, fpfh,
                mask_fpfh_points, map_required_point_idx_to_point_idx);
    } else {
        const int64_t size = input.GetPointPositions().GetLength();
        fpfh = core::Tensor::Zeros({size, 33}, dtype, device);
        pipelines::kernel::ComputeFPFHFeature(
                input.GetPointPositions(), input.GetPointNormals(), p_indices,
                p_distance2, p_counts, fpfh);
    }
    return fpfh;
}

core::Tensor CorrespondencesFromFeatures(const core::Tensor &source_features,
                                         const core::Tensor &target_features,
                                         bool mutual_filter,
                                         float mutual_consistent_ratio) {
    const int num_searches = mutual_filter ? 2 : 1;

    std::array<core::Tensor, 2> features{source_features, target_features};
    std::vector<core::Tensor> corres(num_searches);

    const int kMaxThreads = utility::EstimateMaxThreads();
    const int kOuterThreads = std::min(kMaxThreads, num_searches);
    (void)kOuterThreads;  // Avoids compiler warning if OpenMP is disabled

    // corres[0]: corres_ij, corres[1]: corres_ji
#pragma omp parallel for num_threads(kOuterThreads)
    for (int i = 0; i < num_searches; ++i) {
        core::nns::NearestNeighborSearch nns(features[1 - i],
                                             core::Dtype::Int64);
        nns.KnnIndex();
        auto result = nns.KnnSearch(features[i], 1);

        corres[i] = result.first.View({-1});
    }

    auto corres_ij = corres[0];
    core::Tensor arange_source =
            core::Tensor::Arange(0, source_features.GetLength(), 1,
                                 corres_ij.GetDtype(), corres_ij.GetDevice());

    // Change view for the appending axis
    core::Tensor result_ij =
            arange_source.View({-1, 1}).Append(corres_ij.View({-1, 1}), 1);

    if (!mutual_filter) {
        return result_ij;
    }

    auto corres_ji = corres[1];
    // Mutually consistent
    core::Tensor corres_ii = corres_ji.IndexGet({corres_ij});
    core::Tensor identical = corres_ii.Eq(arange_source);
    core::Tensor result_mutual = corres_ij.IndexGet({identical});
    if (result_mutual.GetLength() >
        mutual_consistent_ratio * arange_source.GetLength()) {
        utility::LogDebug("{:d} correspondences remain after mutual filter",
                          result_mutual.GetLength());
        return arange_source.IndexGet({identical})
                .View({-1, 1})
                .Append(result_mutual.View({-1, 1}), 1);
    }
    // fall back to full correspondences
    utility::LogWarning(
            "Too few correspondences ({:d}) after mutual filter, fall back to "
            "original correspondences.",
            result_mutual.GetLength());
    return result_ij;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
