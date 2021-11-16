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
#include "open3d/core/TensorCheck.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/kernel/Registration.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static void GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        double max_correspondence_distance,
        const core::Tensor &transformation,
        RegistrationResult &result) {
    core::AssertTensorShape(transformation, {4, 4});

    result.transformation_ =
            transformation.To(core::Device("CPU:0"), core::Float64);

    core::Tensor distances, counts;
    std::tie(result.correspondences_, distances, counts) =
            target_nns.HybridSearch(source.GetPointPositions(),
                                    max_correspondence_distance, 1);
    result.correspondences_ = result.correspondences_.To(core::Int64);
    double num_correspondences =
            counts.Sum({0}).To(core::Float64).Item<double>();

    if (num_correspondences == 0) {
        utility::LogWarning(
                "0 correspondence present between the pointclouds. Try "
                "increasing the max_correspondence_distance parameter.");
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
        result.transformation_ =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));
        return;
    }

    // Reduction sum of "distances" for error.
    double squared_error = distances.Sum({0}).To(core::Float64).Item<double>();

    result.fitness_ =
            num_correspondences /
            static_cast<double>(source.GetPointPositions().GetLength());
    result.inlier_rmse_ = std::sqrt(squared_error / num_correspondences);
}

RegistrationResult EvaluateRegistration(const geometry::PointCloud &source,
                                        const geometry::PointCloud &target,
                                        double max_correspondence_distance,
                                        const core::Tensor &transformation) {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation);

    open3d::core::nns::NearestNeighborSearch target_nns(
            target.GetPointPositions());

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "NearestNeighborSearch::HybridSearch: Index is not set.");
    }

    RegistrationResult result(transformation, false);
    GetRegistrationResultAndCorrespondences(source_transformed, target_nns,
                                            max_correspondence_distance,
                                            transformation, result);
    return result;
}

RegistrationResult ICP(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const double max_correspondence_distance,
                       const core::Tensor &init_source_to_target,
                       const TransformationEstimation &estimation,
                       const ICPConvergenceCriteria &criteria,
                       const double voxel_size,
                       const bool save_loss_log) {
    return MultiScaleICP(source, target, {voxel_size}, {criteria},
                         {max_correspondence_distance}, init_source_to_target,
                         estimation, save_loss_log);
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
    core::AssertTensorShape(init_source_to_target, {4, 4});

    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    core::AssertTensorDtype(target.GetPointPositions(), dtype);
    core::AssertTensorDevice(target.GetPointPositions(), device);

    if (dtype == core::Float64 &&
        device.GetType() == core::Device::DeviceType::CUDA) {
        utility::LogDebug(
                "Use Float32 pointcloud for best performance on CUDA device.");
    }
    if (!(criterias.size() == voxel_sizes.size() &&
          criterias.size() == max_correspondence_distances.size())) {
        utility::LogError(
                "Size of criterias, voxel_size, max_correspondence_distances "
                "vectors must be same.");
    }
    if (estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::PointToPlane &&
        (!target.HasPointNormals())) {
        utility::LogError(
                "TransformationEstimationPointToPlane require pre-computed "
                "normal vectors for target PointCloud.");
    }

    // ColoredICP requires pre-computed color_gradients for target points.
    if (estimation.GetTransformationEstimationType() ==
        TransformationEstimationType::ColoredICP) {
        if (!target.HasPointNormals()) {
            utility::LogError(
                    "ColoredICP requires target pointcloud to have normals.");
        }
        if (!target.HasPointColors()) {
            utility::LogError(
                    "ColoredICP requires target pointcloud to have colors.");
        }
        if (!source.HasPointColors()) {
            utility::LogError(
                    "ColoredICP requires source pointcloud to have colors.");
        }
    }

    if (max_correspondence_distances[0] <= 0.0) {
        utility::LogError(
                " Max correspondence distance must be greater than 0, but"
                " got {} in scale: {}.",
                max_correspondence_distances[0], 0);
    }

    for (int64_t i = 1; i < num_iterations; ++i) {
        if (voxel_sizes[i] >= voxel_sizes[i - 1]) {
            utility::LogError(
                    " [ICP] Voxel sizes must be in strictly decreasing order.");
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
        const double &max_correspondence_distance,
        const TransformationEstimation &estimation,
        const int64_t &num_iterations) {
    std::vector<t::geometry::PointCloud> source_down_pyramid(num_iterations);
    std::vector<t::geometry::PointCloud> target_down_pyramid(num_iterations);

    if (voxel_sizes[num_iterations - 1] <= 0) {
        source_down_pyramid[num_iterations - 1] = source.Clone();
        target_down_pyramid[num_iterations - 1] = target;
    } else {
        source_down_pyramid[num_iterations - 1] =
                source.VoxelDownSample(voxel_sizes[num_iterations - 1]);
        target_down_pyramid[num_iterations - 1] =
                target.VoxelDownSample(voxel_sizes[num_iterations - 1]);
    }

    // Computing Color Gradients.
    if (estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::ColoredICP &&
        !target.HasPointAttr("color_gradients")) {
        // `max_correspondence_distance * 2.0` or
        // `voxel_sizes[num_iterations - 1] * 4.0` is an approximation, for
        // `search_radius` in `EstimateColorGradients`. For more control /
        // performance tunning, one may compute and save the `color_gradient`
        // attribute in the target pointcloud manually by calling the function
        // `EstimateColorGradients`, before passing it to the `ICP` function.
        if (voxel_sizes[num_iterations - 1] <= 0) {
            utility::LogWarning(
                    "Use voxel size parameter, for better performance in "
                    "ColoredICP.");
            target_down_pyramid[num_iterations - 1].EstimateColorGradients(
                    30, max_correspondence_distance * 2.0);
        } else {
            target_down_pyramid[num_iterations - 1].EstimateColorGradients(
                    30, voxel_sizes[num_iterations - 1] * 4.0);
        }
    }

    for (int k = num_iterations - 2; k >= 0; k--) {
        source_down_pyramid[k] =
                source_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
        target_down_pyramid[k] =
                target_down_pyramid[k + 1].VoxelDownSample(voxel_sizes[k]);
    }

    return std::make_tuple(source_down_pyramid, target_down_pyramid);
}

static void DoSingleScaleIterationsICP(
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
        const core::Dtype &dtype,
        RegistrationResult &result) {
    for (int j = 0; j < criteria.max_iteration_; j++) {
        GetRegistrationResultAndCorrespondences(
                source.GetPointPositions(), target_nns,
                max_correspondence_distance, transformation, result);

        if (result.fitness_ <= std::numeric_limits<double>::min()) {
            return;
        }

        // Computing Transform between source and target, given
        // correspondences. ComputeTransformation returns {4,4} shaped
        // Float64 transformation tensor on CPU device.
        core::Tensor update =
                estimation
                        .ComputeTransformation(source, target,
                                               result.correspondences_)
                        .To(core::Float64);

        // Multiply the transform to the cumulative transformation (update).
        transformation = update.Matmul(transformation);

        // Apply the transform on source pointcloud.
        source.Transform(update);

        utility::LogDebug(
                "ICP Scale #{:d} Iteration #{:d}: Fitness {:.4f}, RMSE "
                "{:.4f}",
                iteration_idx, j, result.fitness_, result.inlier_rmse_);

        if (result.save_loss_log_) {
            const core::Device host("CPU:0");

            if (iteration_idx == 0 && j == 0) {
                // Initialize attributes for first iteration.
                result.loss_log_["index"] =
                        core::Tensor::Init<int64_t>({{0}}, host);
                result.loss_log_["scale"] =
                        core::Tensor::Init<int64_t>({{0}}, host);
                result.loss_log_["iteration"] =
                        core::Tensor::Init<int64_t>({{0}}, host);
                result.loss_log_["inlier_rmse"] = core::Tensor::Init<double>(
                        {{result.inlier_rmse_}}, host);
                result.loss_log_["fitness"] =
                        core::Tensor::Init<double>({{result.fitness_}}, host);
                result.loss_log_["transformation"] = transformation.To(host);
            } else {
                // Get iteration debug tensors for this iteration.
                core::Tensor local_index = core::Tensor::Init<int64_t>(
                        {{result.loss_log_["index"].GetLength() + 1}}, host);
                core::Tensor local_scale =
                        core::Tensor::Init<int64_t>({{iteration_idx}}, host);
                core::Tensor local_iteration =
                        core::Tensor::Init<int64_t>({{j}}, host);
                core::Tensor local_rmse = core::Tensor::Init<double>(
                        {{result.inlier_rmse_}}, core::Device("CPU:0"));
                core::Tensor local_fitness =
                        core::Tensor::Init<double>({{result.fitness_}}, host);

                // Concatenate the result of this iteration to the existing
                // TensorMap.
                result.loss_log_["index"] = core::Concatenate(
                        {result.loss_log_["index"], local_index}, 0);
                result.loss_log_["scale"] = core::Concatenate(
                        {result.loss_log_["scale"], local_scale}, 0);
                result.loss_log_["iteration"] = core::Concatenate(
                        {result.loss_log_["iteration"], local_iteration}, 0);
                result.loss_log_["inlier_rmse"] = core::Concatenate(
                        {result.loss_log_["inlier_rmse"], local_rmse}, 0);
                result.loss_log_["fitness"] = core::Concatenate(
                        {result.loss_log_["fitness"], local_fitness}, 0);
                result.loss_log_["transformation"] =
                        core::Concatenate({result.loss_log_["transformation"],
                                           transformation.To(host)},
                                          0);
            }
        }

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
}

RegistrationResult MultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criterias,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init_source_to_target,
        const TransformationEstimation &estimation,
        const bool save_loss_log) {
    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});

    const core::Device device = source.GetDevice();
    const core::Dtype dtype = source.GetPointPositions().GetDtype();
    const int64_t num_iterations = int64_t(criterias.size());

    // Asseting input parameters.
    AssertInputMultiScaleICP(source, target, voxel_sizes, criterias,
                             max_correspondence_distances,
                             init_source_to_target, estimation, num_iterations,
                             device, dtype);

    // Initializing point-cloud by down-sampling and computing required
    // attributes.
    std::vector<t::geometry::PointCloud> source_down_pyramid(num_iterations);
    std::vector<t::geometry::PointCloud> target_down_pyramid(num_iterations);
    std::tie(source_down_pyramid, target_down_pyramid) =
            InitializePointCloudPyramidForMultiScaleICP(
                    source, target, voxel_sizes,
                    max_correspondence_distances[num_iterations - 1],
                    estimation, num_iterations);

    // Transformation tensor is always of shape {4,4}, type Float64 on CPU:0.
    core::Tensor transformation =
            init_source_to_target.To(core::Device("CPU:0"), core::Float64);
    RegistrationResult result(transformation, save_loss_log);

    double prev_fitness = 0;
    double prev_inlier_rmse = 0;

    // ---- Iterating over different resolution scale START -------------------
    for (int64_t i = 0; i < num_iterations; ++i) {
        source_down_pyramid[i].Transform(transformation);

        // Initialize Neighbor Search.
        core::nns::NearestNeighborSearch target_nns(
                target_down_pyramid[i].GetPointPositions());
        bool check = target_nns.HybridIndex(max_correspondence_distances[i]);
        if (!check) {
            utility::LogError(
                    "NearestNeighborSearch::HybridSearch: Index is not set.");
        }

        // ICP iterations result for single scale.
        DoSingleScaleIterationsICP(
                source_down_pyramid[i], target_down_pyramid[i], target_nns,
                criterias[i], max_correspondence_distances[i], transformation,
                estimation, i, prev_fitness, prev_inlier_rmse, device, dtype,
                result);

        // To calculate final `fitness` and `inlier_rmse` for the current
        // `transformation` stored in `result`.
        if (i == num_iterations - 1) {
            GetRegistrationResultAndCorrespondences(
                    source_down_pyramid[i], target_nns,
                    max_correspondence_distances[i], transformation, result);
        }

        // No correspondences.
        if (result.fitness_ <= std::numeric_limits<double>::min()) {
            return result;
        }
    }
    // ---- Iterating over different resolution scale END --------------------

    return result;
}

core::Tensor GetInformationMatrix(const geometry::PointCloud &source,
                                  const geometry::PointCloud &target,
                                  const double max_correspondence_distance,
                                  const core::Tensor &transformation) {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});

    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation);

    open3d::core::nns::NearestNeighborSearch target_nns(
            target.GetPointPositions());

    target_nns.HybridIndex(max_correspondence_distance);

    core::Tensor correspondences, distances, counts;
    std::tie(correspondences, distances, counts) =
            target_nns.HybridSearch(source_transformed.GetPointPositions(),
                                    max_correspondence_distance, 1);

    correspondences = correspondences.To(core::Int64);
    int32_t num_correspondences = counts.Sum({0}).Item<int32_t>();

    if (num_correspondences == 0) {
        utility::LogError(
                "0 correspondence present between the pointclouds. Try "
                "increasing the max_correspondence_distance parameter.");
    }

    return kernel::ComputeInformationMatrix(target.GetPointPositions(),
                                            correspondences);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
