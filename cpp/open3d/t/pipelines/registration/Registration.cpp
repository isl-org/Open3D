// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        open3d::core::nns::NearestNeighborSearch &target_nns,
        double max_correspondence_distance,
        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    transformation.AssertShape({4, 4});
    transformation.AssertDtype(dtype);

    core::Tensor transformation_device = transformation.To(device);

    RegistrationResult result(transformation_device);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "[Tensor: EvaluateRegistration: "
                "GetRegistrationResultAndCorrespondences: "
                "NearestNeighborSearch::HybridSearch] "
                "Index is not set.");
    }

    core::Tensor neighbour_indices, squared_distances;
    result.correspondence_set_ = target_nns.HybridSearch(
            source.GetPoints(), max_correspondence_distance, 1);
    neighbour_indices = result.correspondence_set_.first;
    squared_distances = result.correspondence_set_.second;

    core::Tensor valid = neighbour_indices.Ne(-1).Reshape({-1});
    core::Tensor source_indices =
            core::Tensor::Arange(0, source.GetPoints().GetShape()[0], 1,
                                 core::Dtype::Int64, device)
                    .IndexGet({valid});
    // Only take valid indices.
    neighbour_indices = neighbour_indices.IndexGet({valid}).Reshape({-1});
    // Only take valid distances.
    squared_distances = squared_distances.IndexGet({valid});

    // Number of good correspondences (C).
    int num_correspondences = neighbour_indices.GetLength();

    // Reduction sum of "distances" for error.
    double squared_error =
            static_cast<double>(squared_distances.Sum({0}).Item<float>());
    result.fitness_ = static_cast<double>(num_correspondences) /
                      static_cast<double>(source.GetPoints().GetLength());
    result.inlier_rmse_ =
            std::sqrt(squared_error / static_cast<double>(num_correspondences));
    result.transformation_ = transformation;

    return result;
}

RegistrationResult EvaluateRegistration(const geometry::PointCloud &source,
                                        const geometry::PointCloud &target,
                                        double max_correspondence_distance,
                                        const core::Tensor &transformation) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    transformation.AssertShape({4, 4});
    transformation.AssertDtype(dtype);
    core::Tensor transformation_device = transformation.To(device);

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation_device);
    return GetRegistrationResultAndCorrespondences(
            source_transformed, target, target_nns, max_correspondence_distance,
            transformation_device);
}

RegistrationResult RegistrationICP(const geometry::PointCloud &source,
                                   const geometry::PointCloud &target,
                                   double max_correspondence_distance,
                                   const core::Tensor &init,
                                   const TransformationEstimation &estimation,
                                   const ICPConvergenceCriteria &criteria) {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    if (max_correspondence_distance <= 0.0) {
        utility::LogError(
                " Max correspondence distance must be greater than 0, but got "
                "{}.",
                max_correspondence_distance);
    }

    init.AssertShape({4, 4});
    init.AssertDtype(dtype);
    core::Tensor transformation_device = init.To(device);

    open3d::core::nns::NearestNeighborSearch target_nns(target.GetPoints());

    bool check = target_nns.HybridIndex(max_correspondence_distance);
    if (!check) {
        utility::LogError(
                "[Tensor Registration: NearestNeighborSearch::HybridSearch] "
                "Index is not set.");
    }

    geometry::PointCloud source_transformed = source.Clone();
    source_transformed.Transform(transformation_device);

    // TODO: Default constructor absent in RegistrationResult class.
    RegistrationResult result(transformation_device);

    CorrespondenceSet corres;

    double prev_fitness_ = 0;
    double prev_inliner_rmse_ = 0;

    for (int i = 0; i < criteria.max_iteration_; i++) {
        // Get correspondences.
        corres = target_nns.HybridSearch(source_transformed.GetPoints(),
                                         max_correspondence_distance, 1);
        result.correspondence_set_ = corres;

        // Get transformation, squared_error and number of correspondences
        // between source and target points, given the correspondence_set.
        double squared_error = 0;
        int64_t num_correspondences = 0;
        core::Tensor update = estimation.ComputeTransformation(
                source_transformed, target, corres, squared_error,
                num_correspondences);

        // Multiply the transform to the cumulative transformation (update).
        transformation_device = update.Matmul(transformation_device);
        // Apply the transform on source pointcloud.
        source_transformed.Transform(update);
        result.transformation_ = transformation_device;

        // Calculate fitness and inlier_rmse given the squared_error and number
        // of correspondences.
        result.fitness_ = static_cast<double>(num_correspondences) /
                          static_cast<double>(source.GetPoints().GetLength());
        result.inlier_rmse_ = std::sqrt(
                squared_error / static_cast<double>(num_correspondences));

        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}",
                          i + 1, result.fitness_, result.inlier_rmse_);

        // ICPConvergenceCriteria, to terminate iteration.
        if (i != 0 &&
            std::abs(prev_fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(prev_inliner_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }

        prev_fitness_ = result.fitness_;
        prev_inliner_rmse_ = result.inlier_rmse_;
        utility::LogInfo(" Fitness: {}, RMSE: {}", result.fitness_,
                         result.inlier_rmse_);
    }

    utility::LogInfo(" Fitness: {}, RMSE: {}", result.fitness_,
                     result.inlier_rmse_);
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
