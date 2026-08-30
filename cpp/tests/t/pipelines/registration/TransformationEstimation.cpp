// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TransformationEstimationPermuteDevices : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        TransformationEstimation,
        TransformationEstimationPermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

static std::
        tuple<t::geometry::PointCloud, t::geometry::PointCloud, core::Tensor>
        GetTestPointCloudsAndCorrespondences(const core::Dtype& dtype,
                                             const core::Device& device) {
    core::Tensor source_points =
            core::Tensor::Init<double>({{1.15495, 2.40671, 1.15061},
                                        {1.81481, 2.06281, 1.71927},
                                        {0.888322, 2.05068, 2.04879},
                                        {3.78842, 1.70788, 1.30246},
                                        {1.8437, 2.22894, 0.986237},
                                        {2.95706, 2.20180, 0.987878},
                                        {1.72644, 1.24356, 1.93486},
                                        {0.922024, 1.14872, 2.34317},
                                        {3.70293, 1.85134, 1.15357},
                                        {3.06505, 1.30386, 1.55279},
                                        {0.634826, 1.04995, 2.47046},
                                        {1.40107, 1.37469, 1.09687},
                                        {2.93002, 1.96242, 1.48532},
                                        {3.74384, 1.30258, 1.30244}},
                                       device);

    t::geometry::PointCloud source(source_points.To(device, dtype));

    core::Tensor target_points =
            core::Tensor::Init<double>({{2.41766, 2.05397, 1.74994},
                                        {1.37848, 2.19793, 1.66553},
                                        {2.24325, 2.27183, 1.33708},
                                        {3.09898, 1.98482, 1.77401},
                                        {1.81615, 1.48337, 1.49697},
                                        {3.01758, 2.20312, 1.51502},
                                        {2.38836, 1.39096, 1.74914},
                                        {1.30911, 1.4252, 1.37429},
                                        {3.16847, 1.39194, 1.90959},
                                        {1.59412, 1.53304, 1.58040},
                                        {1.34342, 2.19027, 1.30075}},
                                       device);

    core::Tensor target_normals =
            core::Tensor::Init<double>({{-0.00850160, -0.22355, -0.519574},
                                        {0.257463, -0.0738755, -0.698319},
                                        {0.0574301, -0.484248, -0.409929},
                                        {-0.0123503, -0.230172, -0.520720},
                                        {0.355904, -0.142007, -0.720467},
                                        {0.0674038, -0.418757, -0.458602},
                                        {0.226091, 0.258253, -0.874024},
                                        {0.43979, 0.122441, -0.574998},
                                        {0.109144, 0.180992, -0.762368},
                                        {0.273325, 0.292013, -0.903111},
                                        {0.385407, -0.212348, -0.277818}},
                                       device);

    t::geometry::PointCloud target(target_points.To(device, dtype));
    target.SetPointNormals(target_normals.To(device, dtype));

    core::Tensor corres = core::Tensor::Init<int64_t>(
            {10, 1, 1, 3, 2, 5, 9, 7, 5, 8, 7, 7, 5, 8}, device);

    return std::make_tuple(source, target, corres);
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation_p2p;
        double p2p_rmse =
                estimation_p2p.ComputeRMSE(source_pcd, target_pcd, corres);

        EXPECT_NEAR(p2p_rmse, 0.706437, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPoint
                estimation_p2p;

        // Get transform.
        core::Tensor p2p_transform = estimation_p2p.ComputeTransformation(
                source_pcd, target_pcd, corres);
        // Apply transform.
        t::geometry::PointCloud source_transformed_p2p = source_pcd.Clone();
        source_transformed_p2p.Transform(p2p_transform);
        double p2p_rmse_ = estimation_p2p.ComputeRMSE(source_transformed_p2p,
                                                      target_pcd, corres);

        // Compare the new RMSE after transformation.
        EXPECT_NEAR(p2p_rmse_, 0.578255, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSEPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation_p2plane;
        double p2plane_rmse =
                estimation_p2plane.ComputeRMSE(source_pcd, target_pcd, corres);

        EXPECT_NEAR(p2plane_rmse, 0.335499, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices,
       ComputeTransformationPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        t::pipelines::registration::TransformationEstimationPointToPlane
                estimation_p2plane;

        // Get transform.
        core::Tensor p2plane_transform =
                estimation_p2plane.ComputeTransformation(source_pcd, target_pcd,
                                                         corres);
        // Apply transform.
        t::geometry::PointCloud source_transformed_p2plane = source_pcd.Clone();
        source_transformed_p2plane.Transform(p2plane_transform);
        double p2plane_rmse_ = estimation_p2plane.ComputeRMSE(
                source_transformed_p2plane, target_pcd, corres);

        // Compare the new RMSE after transformation.
        EXPECT_NEAR(p2plane_rmse_, 0.601422, 0.0001);
    }
}

TEST_P(TransformationEstimationPermuteDevices, ComputeRMSESymmetric) {
    const core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        const double tolerance = dtype == core::Float32 ? 1e-6 : 1e-12;
        t::geometry::PointCloud source(
                core::Tensor::Init<double>({{1.0, 0.0, 0.0}}, device)
                        .To(device, dtype));
        source.SetPointNormals(
                core::Tensor::Init<double>({{1.0, 0.0, 0.0}}, device)
                        .To(device, dtype));
        t::geometry::PointCloud target(
                core::Tensor::Init<double>({{0.5, std::sqrt(3.0) / 2.0, 0.0}},
                                           device)
                        .To(device, dtype));
        target.SetPointNormals(
                core::Tensor::Init<double>({{0.5, std::sqrt(3.0) / 2.0, 0.0}},
                                           device)
                        .To(device, dtype));
        const core::Tensor correspondences =
                core::Tensor::Init<int64_t>({0}, device);
        t::pipelines::registration::TransformationEstimationSymmetric
                estimation;

        EXPECT_NEAR(estimation.ComputeRMSE(source, target, correspondences),
                    0.0, tolerance);
        target.SetPointNormals(target.GetPointNormals() * -1.0);
        EXPECT_NEAR(estimation.ComputeRMSE(source, target, correspondences),
                    0.0, tolerance);

        target.SetPointPositions(core::Tensor::Zeros({1, 3}, dtype, device));
        target.SetPointNormals(source.GetPointNormals().Clone());
        EXPECT_NEAR(estimation.ComputeRMSE(source, target, correspondences),
                    2.0, tolerance);
        target.SetPointNormals(target.GetPointNormals() * -1.0);
        EXPECT_NEAR(estimation.ComputeRMSE(source, target, correspondences),
                    2.0, tolerance);
    }
}

TEST_P(TransformationEstimationPermuteDevices, ComputeTransformationSymmetric) {
    const core::Device device = GetParam();
    const core::Device cpu("CPU:0");

    Eigen::Matrix4d expected_eigen = Eigen::Matrix4d::Identity();
    expected_eigen.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())
                    .toRotationMatrix();
    expected_eigen.block<3, 1>(0, 3) = Eigen::Vector3d(0.2, -0.1, 0.15);
    const core::Tensor expected =
            core::eigen_converter::EigenMatrixToTensor(expected_eigen);
    const core::Tensor robust_expected =
            core::Tensor::Init<double>({{0.978808528971923, 0.011598608561290,
                                         -0.204448858865149, 0.374573231881982},
                                        {-0.033468146467010, 0.994030976876383,
                                         -0.103837855246761, 0.105509532797345},
                                        {0.202024124262134, 0.108479902699193,
                                         0.973354182159039, -0.112542276940963},
                                        {0.0, 0.0, 0.0, 1.0}},
                                       cpu);

    for (auto dtype : {core::Float32, core::Float64}) {
        const double tolerance = dtype == core::Float32 ? 1e-4 : 1e-8;
        t::geometry::PointCloud source(
                core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                            {1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0},
                                            {1.0, 1.0, 0.0},
                                            {1.0, 0.0, 1.0}},
                                           device)
                        .To(device, dtype));
        source.SetPointNormals(core::Tensor::Init<double>({{1.0, 2.0, 3.0},
                                                           {2.0, -1.0, 1.0},
                                                           {-1.0, 3.0, 2.0},
                                                           {3.0, 1.0, -2.0},
                                                           {-2.0, -1.0, 3.0},
                                                           {1.0, -3.0, 2.0}},
                                                          device)
                                       .To(device, dtype));
        source.NormalizeNormals();

        t::geometry::PointCloud target = source.Clone();
        target.Transform(expected.To(device, dtype));
        const core::Tensor correspondences =
                core::Tensor::Arange(0, 6, 1, core::Int64, device);
        t::pipelines::registration::TransformationEstimationSymmetric
                estimation;
        const auto expect_transform = [&](const core::Tensor& actual) {
            EXPECT_EQ(actual.GetShape(), core::SizeVector({4, 4}));
            EXPECT_EQ(actual.GetDtype(), core::Float64);
            EXPECT_EQ(actual.GetDevice(), cpu);
            EXPECT_TRUE(actual.AllClose(expected, tolerance, tolerance));
        };

        expect_transform(estimation.ComputeTransformation(source, target,
                                                          correspondences));

        const core::Tensor alternating_signs =
                core::Tensor::Init<double>(
                        {{-1.0}, {1.0}, {-1.0}, {1.0}, {-1.0}, {1.0}}, device)
                        .To(device, dtype);
        target.SetPointNormals(target.GetPointNormals() * alternating_signs);
        expect_transform(estimation.ComputeTransformation(source, target,
                                                          correspondences));

        const core::Tensor no_correspondences =
                core::Tensor::Full({6}, -1, core::Int64, device);
        const core::Tensor identity = estimation.ComputeTransformation(
                source, target, no_correspondences);
        EXPECT_EQ(identity.GetDtype(), core::Float64);
        EXPECT_EQ(identity.GetDevice(), cpu);
        EXPECT_TRUE(identity.AllClose(core::Tensor::Eye(4, core::Float64, cpu),
                                      0.0, 0.0));

        const auto expect_direct_methods_throw =
                [&](const t::geometry::PointCloud& checked_source,
                    const t::geometry::PointCloud& checked_target,
                    const core::Tensor& checked_correspondences) {
                    EXPECT_THROW(estimation.ComputeRMSE(
                                         checked_source, checked_target,
                                         checked_correspondences),
                                 std::runtime_error);
                    EXPECT_THROW(estimation.ComputeTransformation(
                                         checked_source, checked_target,
                                         checked_correspondences),
                                 std::runtime_error);
                };

        core::Tensor correspondence_below_range = correspondences.Clone();
        correspondence_below_range[0] = -2;
        expect_direct_methods_throw(source, target, correspondence_below_range);
        core::Tensor correspondence_above_range = correspondences.Clone();
        correspondence_above_range[0] = target.GetPointPositions().GetLength();
        expect_direct_methods_throw(source, target, correspondence_above_range);

        t::geometry::PointCloud malformed_source_normals = source.Clone();
        malformed_source_normals.GetPointNormals() =
                core::Tensor::Zeros({6, 1}, dtype, device);
        expect_direct_methods_throw(malformed_source_normals, target,
                                    correspondences);
        t::geometry::PointCloud malformed_target_normals = target.Clone();
        malformed_target_normals.GetPointNormals() =
                core::Tensor::Zeros({6, 1}, dtype, device);
        expect_direct_methods_throw(source, malformed_target_normals,
                                    correspondences);
        t::geometry::PointCloud malformed_source_positions = source.Clone();
        malformed_source_positions.GetPointPositions() =
                core::Tensor::Zeros({6, 1}, dtype, device);
        expect_direct_methods_throw(malformed_source_positions, target,
                                    correspondences);
        t::geometry::PointCloud malformed_target_positions = target.Clone();
        malformed_target_positions.GetPointPositions() =
                core::Tensor::Zeros({6, 1}, dtype, device);
        expect_direct_methods_throw(source, malformed_target_positions,
                                    correspondences);

        t::geometry::PointCloud robust_source(
                core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                            {1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0},
                                            {1.0, 1.0, 0.0},
                                            {1.0, 0.0, 1.0},
                                            {0.0, 1.0, 1.0},
                                            {1.0, 1.0, 1.0},
                                            {2.0, -1.0, 0.5},
                                            {-0.5, 1.5, 2.0}},
                                           device)
                        .To(device, dtype));
        robust_source.SetPointNormals(
                core::Tensor::Init<double>({{1.0, 2.0, 3.0},
                                            {2.0, -1.0, 1.0},
                                            {-1.0, 3.0, 2.0},
                                            {3.0, 1.0, -2.0},
                                            {-2.0, -1.0, 3.0},
                                            {1.0, -3.0, 2.0},
                                            {-3.0, 2.0, 1.0},
                                            {2.0, 3.0, -1.0},
                                            {1.0, 1.0, -2.0},
                                            {-2.0, 1.0, -3.0}},
                                           device)
                        .To(device, dtype));
        robust_source.NormalizeNormals();
        t::geometry::PointCloud robust_target = robust_source.Clone();
        robust_target.Transform(expected.To(device, dtype));
        const core::Tensor robust_noise =
                core::Tensor::Init<double>({{0.02, -0.01, 0.0},
                                            {-0.03, 0.02, 0.01},
                                            {0.0, 0.04, -0.02},
                                            {0.01, -0.03, 0.03},
                                            {-0.04, 0.0, 0.02},
                                            {0.03, 0.01, -0.04},
                                            {-0.02, -0.02, 0.03},
                                            {0.04, -0.03, -0.01},
                                            {0.85, -0.55, 0.45},
                                            {-0.65, 0.70, -0.50}},
                                           device)
                        .To(device, dtype);
        robust_target.SetPointPositions(robust_target.GetPointPositions() +
                                        robust_noise);
        const core::Tensor robust_correspondences =
                core::Tensor::Arange(0, 10, 1, core::Int64, device);
        const t::pipelines::registration::TransformationEstimationSymmetric
                robust_estimation(t::pipelines::registration::RobustKernel(
                        t::pipelines::registration::RobustKernelMethod::
                                CauchyLoss,
                        0.5, 1.0));
        // The independently calculated centered-residual-weight transform
        // differs by more than 0.12, making this a raw-weight discriminator.
        EXPECT_TRUE(robust_estimation
                            .ComputeTransformation(robust_source, robust_target,
                                                   robust_correspondences)
                            .AllClose(robust_expected, tolerance, tolerance));

        const core::Tensor source_normals = source.GetPointNormals().Clone();
        source.RemovePointAttr("normals");
        EXPECT_THROW(estimation.ComputeRMSE(source, target, correspondences),
                     std::runtime_error);
        EXPECT_THROW(estimation.ComputeTransformation(source, target,
                                                      correspondences),
                     std::runtime_error);

        source.SetPointNormals(source_normals);
        target.RemovePointAttr("normals");
        EXPECT_THROW(estimation.ComputeRMSE(source, target, correspondences),
                     std::runtime_error);
        EXPECT_THROW(estimation.ComputeTransformation(source, target,
                                                      correspondences),
                     std::runtime_error);
    }
}

}  // namespace tests
}  // namespace open3d
