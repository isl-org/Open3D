// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/Feature.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace registration {

core::Tensor ComputeFPFHFeature(const geometry::PointCloud &input,
                                const utility::optional<int> max_nn,
                                const utility::optional<double> radius) {
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

    // Compute nearest neighbors and squared distances.
    core::Tensor indices, distance2, counts;
    core::nns::NearestNeighborSearch tree(input.GetPointPositions(),
                                          core::Int32);
    if (radius.has_value() && max_nn.has_value()) {
        bool check = tree.HybridIndex(radius.value());
        if (!check) {
            utility::LogError("Building HybridIndex failed.");
        }
        std::tie(indices, distance2, counts) = tree.HybridSearch(
                input.GetPointPositions(), radius.value(), max_nn.value());
        utility::LogDebug(
                "Use HybridSearch [max_nn: {} | radius {}] for computing FPFH "
                "feature.",
                max_nn.value(), radius.value());
    } else if (!radius.has_value() && max_nn.has_value()) {
        bool check = tree.KnnIndex();
        if (!check) {
            utility::LogError("Building KnnIndex failed.");
        }
        std::tie(indices, distance2) =
                tree.KnnSearch(input.GetPointPositions(), max_nn.value());

        // Make counts full with min(max_nn, num_points).
        const int fill_value =
                max_nn.value() > num_points ? num_points : max_nn.value();
        counts = core::Tensor::Full({num_points}, fill_value, core::Int32,
                                    device);
        utility::LogDebug(
                "Use KNNSearch  [max_nn: {}] for computing FPFH feature.",
                max_nn.value());
    } else if (radius.has_value() && !max_nn.has_value()) {
        bool check = tree.FixedRadiusIndex(radius.value());
        if (!check) {
            utility::LogError("Building RadiusIndex failed.");
        }
        std::tie(indices, distance2, counts) = tree.FixedRadiusSearch(
                input.GetPointPositions(), radius.value());
        utility::LogDebug(
                "Use RadiusSearch [radius: {}] for computing FPFH feature.",
                radius.value());
    } else {
        utility::LogError("Both max_nn and radius are none.");
    }

    const int64_t size = input.GetPointPositions().GetLength();

    core::Tensor fpfh = core::Tensor::Zeros({size, 33}, dtype, device);
    pipelines::kernel::ComputeFPFHFeature(input.GetPointPositions(),
                                          input.GetPointNormals(), indices,
                                          distance2, counts, fpfh);
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
