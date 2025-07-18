// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include "open3d/core/TensorCheck.h"
#include "open3d/t/pipelines/kernel/Registration.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

static void AssertValidCorrespondences(
        const core::Tensor &correspondence_indices,
        const core::Tensor &source_points) {
    core::AssertTensorDtype(correspondence_indices, core::Int64);
    core::AssertTensorDevice(correspondence_indices, source_points.GetDevice());

    if (correspondence_indices.GetShape() !=
                core::SizeVector({source_points.GetLength(), 1}) &&
        correspondence_indices.GetShape() !=
                core::SizeVector({source_points.GetLength()})) {
        utility::LogError(
                "Correspondences must be of same length as source point-cloud "
                "positions. Expected correspondences of shape {} or {}, but "
                "got {}.",
                core::SizeVector({source_points.GetLength()}).ToString(),
                core::SizeVector({source_points.GetLength(), 1}).ToString(),
                correspondence_indices.GetShape().ToString());
    }
}

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    core::Tensor valid = correspondences.Ne(-1).Reshape({-1});
    core::Tensor neighbour_indices =
            correspondences.IndexGet({valid}).Reshape({-1});
    core::Tensor source_points_indexed =
            source.GetPointPositions().IndexGet({valid});
    core::Tensor target_points_indexed =
            target.GetPointPositions().IndexGet({neighbour_indices});

    core::Tensor error_t = (source_points_indexed - target_points_indexed);
    error_t.Mul_(error_t);
    double error = error_t.Sum({0, 1}).To(core::Float64).Item<double>();
    return std::sqrt(error /
                     static_cast<double>(neighbour_indices.GetLength()));
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const core::Tensor &current_transform,
        const std::size_t iteration) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    core::Tensor R, t;
    // Get tuple of Rotation {3, 3} and Translation {3} of type Float64.
    std::tie(R, t) = pipelines::kernel::ComputeRtPointToPoint(
            source.GetPointPositions(), target.GetPointPositions(),
            correspondences);

    // Get rigid transformation tensor of {4, 4} of type Float64 on CPU:0
    // device, from rotation {3, 3} and translation {3}.
    return t::pipelines::kernel::RtToTransformation(R, t);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }

    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    core::Tensor valid = correspondences.Ne(-1).Reshape({-1});
    core::Tensor neighbour_indices =
            correspondences.IndexGet({valid}).Reshape({-1});
    core::Tensor source_points_indexed =
            source.GetPointPositions().IndexGet({valid});
    core::Tensor target_points_indexed =
            target.GetPointPositions().IndexGet({neighbour_indices});
    core::Tensor target_normals_indexed =
            target.GetPointNormals().IndexGet({neighbour_indices});

    core::Tensor error_t = (source_points_indexed - target_points_indexed)
                                   .Mul_(target_normals_indexed);
    error_t.Mul_(error_t);
    double error = error_t.Sum({0, 1}).To(core::Float64).Item<double>();
    return std::sqrt(error /
                     static_cast<double>(neighbour_indices.GetLength()));
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const core::Tensor &current_transform,
        const std::size_t iteration) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDtype(target.GetPointNormals(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    // Get pose {6} of type Float64.
    core::Tensor pose = pipelines::kernel::ComputePosePointToPlane(
            source.GetPointPositions(), target.GetPointPositions(),
            target.GetPointNormals(), correspondences, this->kernel_);

    // Get rigid transformation tensor of {4, 4} of type Float64 on CPU:0
    // device, from pose {6}.
    return pipelines::kernel::PoseToTransformation(pose);
}

double TransformationEstimationForColoredICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointColors() || !source.HasPointColors()) {
        utility::LogError(
                "Source and/or Target pointcloud missing colors attribute.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }
    if (!target.HasPointAttr("color_gradients")) {
        utility::LogError(
                "Target pointcloud missing color_gradients attribute.");
    }

    const core::Device device = source.GetPointPositions().GetDevice();
    core::AssertTensorDevice(target.GetPointPositions(), device);

    const core::Dtype dtype = source.GetPointPositions().GetDtype();
    core::AssertTensorDtype(source.GetPointColors(), dtype);
    core::AssertTensorDtype(target.GetPointPositions(), dtype);
    core::AssertTensorDtype(target.GetPointNormals(), dtype);
    core::AssertTensorDtype(target.GetPointAttr("color_gradients"), dtype);
    core::AssertTensorDtype(correspondences, core::Int64);
    core::AssertTensorDevice(correspondences,
                             source.GetPointPositions().GetDevice());
    core::AssertTensorShape(correspondences,
                            {source.GetPointPositions().GetLength()});

    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);

    core::Tensor valid = correspondences.Ne(-1).Reshape({-1});
    core::Tensor neighbour_indices =
            correspondences.IndexGet({valid}).Reshape({-1});

    // vs - source points (or vertices)
    // vt - target points
    // nt - target normals
    // cs - source colors
    // ct - target colors
    // dit - target color gradients
    // is - source intensity
    // it - target intensity
    // vs_proj - source points projection
    // is_proj - source intensity projection

    core::Tensor vs = source.GetPointPositions().IndexGet({valid});
    core::Tensor cs = source.GetPointColors().IndexGet({valid});

    core::Tensor vt = target.GetPointPositions().IndexGet({neighbour_indices});
    core::Tensor nt = target.GetPointNormals().IndexGet({neighbour_indices});
    core::Tensor ct = target.GetPointColors().IndexGet({neighbour_indices});
    core::Tensor dit = target.GetPointAttr("color_gradients")
                               .IndexGet({neighbour_indices});

    // vs_proj = vs - (vs - vt).dot(nt) * nt
    // d = (vs - vt).dot(nt)
    const core::Tensor d = (vs - vt).Mul(nt).Sum({1});
    core::Tensor vs_proj = vs - d.Mul(nt);

    core::Tensor is = cs.Mean({1});
    core::Tensor it = ct.Mean({1});

    // is_proj = (dit.dot(vs_proj - vt)) + it
    core::Tensor is_proj = (dit.Mul(vs_proj - vt)).Sum({1}).Add(it);

    core::Tensor residual_geometric = d.Mul(sqrt_lambda_geometric).Sum({1});
    core::Tensor sq_residual_geometric =
            residual_geometric.Mul(residual_geometric);
    core::Tensor residual_photometric =
            (is - is_proj).Mul(sqrt_lambda_photometric).Sum({1});
    core::Tensor sq_residual_photometric =
            residual_photometric.Mul(residual_photometric);

    double residual = sq_residual_geometric.Add_(sq_residual_photometric)
                              .Sum({0})
                              .To(core::Float64)
                              .Item<double>();

    return residual;
}

core::Tensor TransformationEstimationForColoredICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const core::Tensor &current_transform,
        const std::size_t iteration) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointColors() || !source.HasPointColors()) {
        utility::LogError(
                "Source and/or Target pointcloud missing colors attribute.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }
    if (!target.HasPointAttr("color_gradients")) {
        utility::LogError(
                "Target pointcloud missing color_gradients attribute.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    const core::Dtype dtype = source.GetPointPositions().GetDtype();

    core::AssertTensorDtype(source.GetPointColors(), dtype);
    core::AssertTensorDtype(target.GetPointPositions(), dtype);
    core::AssertTensorDtype(target.GetPointNormals(), dtype);
    core::AssertTensorDtype(target.GetPointColors(), dtype);
    core::AssertTensorDtype(target.GetPointAttr("color_gradients"), dtype);

    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    // Get pose {6} of type Float64 from correspondences indexed source and
    // target point cloud.
    core::Tensor pose = pipelines::kernel::ComputePoseColoredICP(
            source.GetPointPositions(), source.GetPointColors(),
            target.GetPointPositions(), target.GetPointNormals(),
            target.GetPointColors(), target.GetPointAttr("color_gradients"),
            correspondences, this->kernel_, this->lambda_geometric_);

    // Get transformation {4,4} of type Float64 from pose {6}.
    core::Tensor transform = pipelines::kernel::PoseToTransformation(pose);

    return transform;
}

double TransformationEstimationForDopplerICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences) const {
    if (!target.HasPointPositions() || !source.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }

    core::AssertTensorDtype(target.GetPointPositions(),
                            source.GetPointPositions().GetDtype());
    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    core::Tensor valid = correspondences.Ne(-1).Reshape({-1});
    core::Tensor neighbour_indices =
            correspondences.IndexGet({valid}).Reshape({-1});
    core::Tensor source_points_indexed =
            source.GetPointPositions().IndexGet({valid});
    core::Tensor target_points_indexed =
            target.GetPointPositions().IndexGet({neighbour_indices});
    core::Tensor target_normals_indexed =
            target.GetPointNormals().IndexGet({neighbour_indices});

    core::Tensor error_t = (source_points_indexed - target_points_indexed)
                                   .Mul_(target_normals_indexed);
    error_t.Mul_(error_t);
    double error = error_t.Sum({0, 1}).To(core::Float64).Item<double>();
    return std::sqrt(error /
                     static_cast<double>(neighbour_indices.GetLength()));
}

core::Tensor TransformationEstimationForDopplerICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const core::Tensor &current_transform,
        const std::size_t iteration) const {
    if (!source.HasPointPositions() || !target.HasPointPositions()) {
        utility::LogError("Source and/or Target pointcloud is empty.");
    }
    if (!target.HasPointNormals()) {
        utility::LogError("Target pointcloud missing normals attribute.");
    }
    if (!source.HasPointAttr("dopplers")) {
        utility::LogError("Source pointcloud missing dopplers attribute.");
    }
    if (!source.HasPointAttr("directions")) {
        utility::LogError("Source pointcloud missing directions attribute.");
    }

    core::AssertTensorDtypes(source.GetPointPositions(),
                             {core::Float64, core::Float32});
    const core::Dtype dtype = source.GetPointPositions().GetDtype();

    core::AssertTensorDtype(target.GetPointPositions(), dtype);
    core::AssertTensorDtype(target.GetPointNormals(), dtype);
    core::AssertTensorDtype(source.GetPointAttr("dopplers"), dtype);
    core::AssertTensorDtype(source.GetPointAttr("directions"), dtype);

    core::AssertTensorDevice(target.GetPointPositions(), source.GetDevice());

    AssertValidCorrespondences(correspondences, source.GetPointPositions());

    // Get pose {6} of type Float64 from correspondences indexed source
    // and target point cloud.
    core::Tensor pose = pipelines::kernel::ComputePoseDopplerICP(
            source.GetPointPositions(), source.GetPointAttr("dopplers"),
            source.GetPointAttr("directions"), target.GetPointPositions(),
            target.GetPointNormals(), correspondences, current_transform,
            this->transform_vehicle_to_sensor_, iteration, this->period_,
            this->lambda_doppler_, this->reject_dynamic_outliers_,
            this->doppler_outlier_threshold_,
            this->outlier_rejection_min_iteration_,
            this->geometric_robust_loss_min_iteration_,
            this->doppler_robust_loss_min_iteration_, this->geometric_kernel_,
            this->doppler_kernel_);

    // Get transformation {4,4} of type Float64 from pose {6}.
    core::Tensor transform = pipelines::kernel::PoseToTransformation(pose);

    return transform;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
