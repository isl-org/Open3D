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

#include "open3d/t/pipelines/registration/Registration.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        double max_correspondence_distance,
        const core::Tensor &transformation) {
    transformation.AssertShape({4, 4});

    core::Tensor transformation_host =
            transformation.To(core::Device("CPU:0"), core::Dtype::Float64);

    RegistrationResult result(transformation_host);

    core::Tensor distances, counts;
    std::tie(result.correspondences_, distances, counts) =
            target_nns.HybridSearch(source.GetPoints(),
                                    max_correspondence_distance, 1);

    double num_correspondences =
            counts.Sum({0}).To(core::Dtype::Float64).Item<double>();

    // Reduction sum of "distances" for error.
    double squared_error =
            distances.Sum({0}).To(core::Dtype::Float64).Item<double>();

    result.fitness_ = num_correspondences /
                      static_cast<double>(source.GetPoints().GetLength());
    result.inlier_rmse_ = std::sqrt(squared_error / num_correspondences);

    return result;
}

RegistrationResult EvaluateRegistration(const geometry::PointCloud &source,
                                        const geometry::PointCloud &target,
                                        double max_correspondence_distance,
                                        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = source.GetPoints().GetDtype();

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation.To(device, dtype));

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "NearestNeighborSearch::HybridSearch: Index is not set.");
    }

    return GetRegistrationResultAndCorrespondences(
            source_transformed, target_nns, max_correspondence_distance,
            transformation);
}

RegistrationResult RegistrationICP(const geometry::PointCloud &source,
                                   const geometry::PointCloud &target,
                                   double max_correspondence_distance,
                                   const core::Tensor &init_source_to_target,
                                   const TransformationEstimation &estimation,
                                   const ICPConvergenceCriteria &criteria) {
    return RegistrationMultiScaleICP(source, target, {-1}, {criteria},
                                     {max_correspondence_distance},
                                     init_source_to_target, estimation);
}

static void AssertInputMultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criterias,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init_source_to_target,
        const TransformationEstimation &estimation,
        const int64_t &num_iterations,
        const core::Device &device,
        const core::Dtype &dtype) {
    init_source_to_target.AssertShape({4, 4});

    if (target.GetPoints().GetDtype() != dtype) {
        utility::LogError(
                "Target Pointcloud dtype {} != Source Pointcloud's dtype {}.",
                target.GetPoints().GetDtype().ToString(), dtype.ToString());
    }
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    if (dtype == core::Dtype::Float64 &&
        device.GetType() == core::Device::DeviceType::CUDA) {
        utility::LogDebug(
                "Use Float32 pointcloud for best performance on CUDA device.");
    }
    if (!(criterias.size() == voxel_sizes.size() &&
          criterias.size() == max_correspondence_distances.size())) {
        utility::LogError(
                " [RegistrationMultiScaleICP]: Size of criterias, voxel_size,"
                " max_correspondence_distances vectors must be same.");
    }
    if (estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::PointToPlane &&
        (!target.HasPointNormals())) {
        utility::LogError(
                "TransformationEstimationPointToPlane require pre-computed "
                "normal vectors for target PointCloud.");
    }

    if (estimation.GetTransformationEstimationType() ==
        TransformationEstimationType::ColoredICP) {
        utility::LogError("Tensor PointCloud ColoredICP is not Implemented.");
    }

    if (max_correspondence_distances[0] <= 0.0) {
        utility::LogError(
                " Max correspondence distance must be greater than 0, but"
                " got {} in scale: {}.",
                max_correspondence_distances[0], 0);
    }

    for (int64_t i = 1; i < num_iterations; i++) {
        if (voxel_sizes[i] >= voxel_sizes[i - 1]) {
            utility::LogError(
                    " [MultiScaleICP] Voxel sizes must be in strictly "
                    "decreasing order.");
        }
        if (max_correspondence_distances[i] <= 0.0) {
            utility::LogError(
                    " Max correspondence distance must be greater than 0, but"
                    " got {} in scale: {}.",
                    max_correspondence_distances[i], i);
        }
    }
}

static std::tuple<std::vector<t::geometry::PointCloud>,
                  std::vector<t::geometry::PointCloud>>
InitializePointCloudPyramidForMultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const TransformationEstimation &estimation,
        const int64_t &num_iterations) {
    std::vector<t::geometry::PointCloud> source_down_pyramid(num_iterations);
    std::vector<t::geometry::PointCloud> target_down_pyramid(num_iterations);

    if (voxel_sizes[num_iterations - 1] == -1) {
        source_down_pyramid[num_iterations - 1] = source.Clone();
        target_down_pyramid[num_iterations - 1] = target;
    } else {
        source_down_pyramid[num_iterations - 1] =
                source.VoxelDownSample(voxel_sizes[num_iterations - 1]);
        target_down_pyramid[num_iterations - 1] =
                target.VoxelDownSample(voxel_sizes[num_iterations - 1]);
    }

    // TODO(@rishabh): Estimate Color-Gradient here, for ColoredICP.
    if (estimation.GetTransformationEstimationType() ==
        TransformationEstimationType::ColoredICP) {
        utility::LogError(" ColoredICP requires pre-computed color-gradients.");
    }

    for (int k = num_iterations - 2; k >= 0; k--) {
        source_down_pyramid[k] =
                source_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
        target_down_pyramid[k] =
                target_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
    }

    return std::make_tuple(source_down_pyramid, target_down_pyramid);
}

static RegistrationResult DoSingleScaleIterationsICP(
        geometry::PointCloud &source,
        const geometry::PointCloud &target,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        const ICPConvergenceCriteria &criteria,
        const double &max_correspondence_distance,
        core::Tensor &transformation,
        const TransformationEstimation &estimation,
        const int &iteration_idx,
        double &prev_fitness,
        double &prev_inlier_rmse,
        const core::Device &device,
        const core::Dtype &dtype) {
    RegistrationResult result;
    for (int j = 0; j < criteria.max_iteration_; j++) {
        result = GetRegistrationResultAndCorrespondences(
                source.GetPoints(), target_nns, max_correspondence_distance,
                transformation);

        // Computing Transform between source and target, given
        // correspondences. ComputeTransformation returns {4,4} shaped
        // Float64 transformation tensor on CPU device.
        core::Tensor update =
                estimation
                        .ComputeTransformation(source, target,
                                               result.correspondences_)
                        .To(core::Dtype::Float64);

        // Multiply the transform to the cumulative transformation (update).
        transformation = update.Matmul(transformation);

        // Apply the transform on source pointcloud.
        source.Transform(update.To(device, dtype));

        utility::LogDebug(
                " ICP Scale #{:d} Iteration #{:d}: Fitness {:.4f}, RMSE "
                "{:.4f}",
                iteration_idx + 1, j, result.fitness_, result.inlier_rmse_);

        // ICPConvergenceCriteria, to terminate iteration.
        if (j != 0 &&
            std::abs(prev_fitness - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(prev_inlier_rmse - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }

        prev_fitness = result.fitness_;
        prev_inlier_rmse = result.inlier_rmse_;
    }

    return result;
}

RegistrationResult RegistrationMultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criterias,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init_source_to_target,
        const TransformationEstimation &estimation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = source.GetPoints().GetDtype();
    int64_t num_iterations = int64_t(criterias.size());

    AssertInputMultiScaleICP(source, target, voxel_sizes, criterias,
                             max_correspondence_distances,
                             init_source_to_target, estimation, num_iterations,
                             device, dtype);

    std::vector<t::geometry::PointCloud> source_down_pyramid(num_iterations);
    std::vector<t::geometry::PointCloud> target_down_pyramid(num_iterations);
    std::tie(source_down_pyramid, target_down_pyramid) =
            InitializePointCloudPyramidForMultiScaleICP(
                    source, target, voxel_sizes, estimation, num_iterations);

    // Transformation tensor is always of shape {4,4}, type Float64 on CPU:0.
    core::Tensor transformation = init_source_to_target.To(
            core::Device("CPU:0"), core::Dtype::Float64);
    RegistrationResult result(transformation);

    double prev_fitness = 0;
    double prev_inlier_rmse = 0;

    // ---- Iterating over different resolution scale START -------------------
    for (int64_t i = 0; i < num_iterations; i++) {
        source_down_pyramid[i].Transform(transformation.To(device, dtype));

        // Initialize Neighbor Search.
        core::nns::NearestNeighborSearch target_nns(
                target_down_pyramid[i].GetPoints());
        bool check = target_nns.HybridIndex(max_correspondence_distances[i]);
        if (!check) {
            utility::LogError(
                    "NearestNeighborSearch::HybridSearch: Index is not set.");
        }

        // ICP iterations result for single scale.
        result = DoSingleScaleIterationsICP(
                source_down_pyramid[i], target_down_pyramid[i], target_nns,
                criterias[i], max_correspondence_distances[i], transformation,
                estimation, i, prev_fitness, prev_inlier_rmse, device, dtype);

        // To calculate final `fitness` and `inlier_rmse` for the current
        // `transformation` stored in `result`.
        if (i == num_iterations - 1) {
            result = GetRegistrationResultAndCorrespondences(
                    source_down_pyramid[i], target_nns,
                    max_correspondence_distances[i], transformation);
        }
    }
    // ---- Iterating over different resolution scale END ---------------------

    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
