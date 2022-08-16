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
                core::Tensor({0}, core::Bool, input.GetDevice()));
    }

    double salient_radius_d = salient_radius.value();
    double non_max_radius_d = non_max_radius.value();
    if (!salient_radius.has_value() || !non_max_radius.has_value() ||
        salient_radius.value() <= 0.0 || non_max_radius.value() <= 0.0) {
        const double resolution = ComputePointCloudResolution(input);
        salient_radius_d = 6 * resolution;
        non_max_radius_d = 4 * resolution;
        utility::LogDebug(
                "Computed salient_radius = {}, non_max_radius = {} from input "
                "pointcloud",
                salient_radius_d, non_max_radius_d);
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

}  // namespace keypoint
}  // namespace geometry
}  // namespace t
}  // namespace open3d