// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/t/geometry/Keypoint.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace keypoint {

double ComputePointCloudResolution(const PointCloud& points) {
    core::nns::NearestNeighborSearch nns(points.GetPointPositions(),
                                         core::Int32);
    nns.KnnIndex();
    core::Tensor indices, distances2;
    std::tie(indices, distances2) =
            nns.KnnSearch(points.GetPointPositions(), 2);

    return distances2.Sum({0})
            .GetItem(core::TensorKey::Index(1))
            .To(core::Float64)
            .Item<double>();
}

std::tuple<PointCloud, core::Tensor> ComputeISSKeypoints(
        const PointCloud& input,
        const utility::optional<double> salient_radius,
        const utility::optional<double> non_max_radius,
        double gamma_21,
        double gamma_32,
        int min_neighbors) {
    core::AssertTensorDtypes(input.GetPointPositions(),
                             {core::Float32, core::Float64});
    if (!input.HasPointPositions()) {
        utility::LogDebug("Input PointCloud is empty!");
        return std::make_tuple(
                PointCloud(input.GetDevice()),
                core::Tensor({0}, core::Int64, input.GetDevice()));
    }

    double salient_radius_d, non_max_radius_d;
    if (!salient_radius.has_value() || !non_max_radius.has_value()) {
        const double resolution = ComputePointCloudResolution(input);
        salient_radius_d = 6 * resolution;
        non_max_radius_d = 4 * resolution;
        utility::LogDebug(
                "Computed salient_radius = {}, non_max_radius = {} from input "
                "pointcloud",
                salient_radius_d, non_max_radius_d);
    } else {
        salient_radius_d = salient_radius.value();
        non_max_radius_d = non_max_radius.value();
    }

    core::Tensor keypoints_mask =
            core::Tensor::Zeros({input.GetPointPositions().GetLength()},
                                core::Bool, input.GetDevice());
    if (input.GetDevice().IsCPU()) {
        kernel::pointcloud::ComputeISSKeypointsCPU(
                input.GetPointPositions(), salient_radius_d, non_max_radius_d,
                gamma_21, gamma_32, min_neighbors, keypoints_mask);
    } else if (input.GetDevice().IsCUDA()) {
        CUDA_CALL(kernel::pointcloud::ComputeISSKeypointsCUDA,
                  input.GetPointPositions(), salient_radius_d, non_max_radius_d,
                  gamma_21, gamma_32, min_neighbors, keypoints_mask);
    } else {
        utility::LogError("Unimplemented device");
    }

    return std::make_tuple(input.SelectByMask(keypoints_mask), keypoints_mask);
}

std::tuple<PointCloud, core::Tensor> ComputeBoundaryPoints(
        const PointCloud& input,
        const utility::optional<int> max_nn,
        const utility::optional<double> radius,
        double angle_threshold) {
    core::AssertTensorDtypes(input.GetPointPositions(),
                             {core::Float32, core::Float64});
    if (!input.HasPointPositions()) {
        utility::LogDebug("Input PointCloud is empty.");
        return std::make_tuple(
                PointCloud(input.GetDevice()),
                core::Tensor({0}, core::Int64, input.GetDevice()));
    }
    if (!input.HasPointNormals()) {
        utility::LogError("Input PointCloud has no normals.");
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
                "Use HybridSearch [max_nn: {} | radius {}] for computing "
                "boundary points.",
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
                "Use KNNSearch [max_nn: {}] for computing boundary points.",
                max_nn.value());
    } else if (radius.has_value() && !max_nn.has_value()) {
        bool check = tree.FixedRadiusIndex(radius.value());
        if (!check) {
            utility::LogError("Building RadiusIndex failed.");
        }
        std::tie(indices, distance2, counts) = tree.FixedRadiusSearch(
                input.GetPointPositions(), radius.value());
        utility::LogDebug(
                "Use RadiusSearch [radius: {}] for computing boundary points.",
                radius.value());
    } else {
        utility::LogError("Both max_nn and radius are none.");
    }

    core::Tensor mask = core::Tensor::Zeros({num_points}, core::Bool, device);
    return std::make_tuple(input.SelectByMask(mask), mask.NonZero().Flatten());
}

}  // namespace keypoint
}  // namespace geometry
}  // namespace t
}  // namespace open3d