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

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
