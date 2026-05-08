// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TransformationEstimationPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TransformationEstimation,
                         TransformationEstimationPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

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
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        // Add normals to source point cloud (required for symmetric ICP)
        core::Tensor source_normals =
                core::Tensor::Init<double>({{-0.1, -0.2, -0.8},
                                            {0.2, -0.1, -0.7},
                                            {0.05, -0.4, -0.6},
                                            {-0.01, -0.3, -0.5},
                                            {0.3, -0.1, -0.7},
                                            {0.06, -0.4, -0.4},
                                            {0.2, 0.2, -0.8},
                                            {0.4, 0.1, -0.5},
                                            {0.1, 0.1, -0.7},
                                            {0.2, 0.3, -0.9},
                                            {0.3, -0.2, -0.2},
                                            {0.1, 0.1, -0.6},
                                            {0.05, -0.4, -0.4},
                                            {0.1, 0.2, -0.7}},
                                           device);
        source_pcd.SetPointNormals(source_normals.To(device, dtype));

        t::pipelines::registration::TransformationEstimationSymmetric
                estimation_symmetric;
        double symmetric_rmse = estimation_symmetric.ComputeRMSE(
                source_pcd, target_pcd, corres);

        // Symmetric RMSE should be positive and finite
        EXPECT_GT(symmetric_rmse, 0.0);
        EXPECT_TRUE(std::isfinite(symmetric_rmse));
    }
}

TEST_P(TransformationEstimationPermuteDevices, ComputeTransformationSymmetric) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        // Add normals to source point cloud (required for symmetric ICP)
        core::Tensor source_normals =
                core::Tensor::Init<double>({{-0.1, -0.2, -0.8},
                                            {0.2, -0.1, -0.7},
                                            {0.05, -0.4, -0.6},
                                            {-0.01, -0.3, -0.5},
                                            {0.3, -0.1, -0.7},
                                            {0.06, -0.4, -0.4},
                                            {0.2, 0.2, -0.8},
                                            {0.4, 0.1, -0.5},
                                            {0.1, 0.1, -0.7},
                                            {0.2, 0.3, -0.9},
                                            {0.3, -0.2, -0.2},
                                            {0.1, 0.1, -0.6},
                                            {0.05, -0.4, -0.4},
                                            {0.1, 0.2, -0.7}},
                                           device);
        source_pcd.SetPointNormals(source_normals.To(device, dtype));

        t::pipelines::registration::TransformationEstimationSymmetric
                estimation_symmetric;

        // Compute initial RMSE
        double initial_rmse = estimation_symmetric.ComputeRMSE(
                source_pcd, target_pcd, corres);
        (void)initial_rmse;  // Suppress unused variable warning

        // Get transform
        core::Tensor symmetric_transform =
                estimation_symmetric.ComputeTransformation(source_pcd,
                                                           target_pcd, corres);

        // Verify transformation is 4x4 matrix
        EXPECT_EQ(symmetric_transform.GetShape(), core::SizeVector({4, 4}));
        EXPECT_EQ(symmetric_transform.GetDtype(), core::Float64);

        // Apply transform
        t::geometry::PointCloud source_transformed_symmetric =
                source_pcd.Clone();
        source_transformed_symmetric.Transform(symmetric_transform);
        double final_rmse = estimation_symmetric.ComputeRMSE(
                source_transformed_symmetric, target_pcd, corres);

        // Final RMSE should be finite and potentially lower than initial
        EXPECT_TRUE(std::isfinite(final_rmse));
        EXPECT_GE(final_rmse, 0.0);
    }
}

TEST_P(TransformationEstimationPermuteDevices, SymmetricICPDeviceConsistency) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // Create simple test data for device consistency testing
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        // Add normals to source
        core::Tensor source_normals =
                core::Tensor::Init<double>({{-0.1, -0.2, -0.8},
                                            {0.2, -0.1, -0.7},
                                            {0.05, -0.4, -0.6},
                                            {-0.01, -0.3, -0.5},
                                            {0.3, -0.1, -0.7},
                                            {0.06, -0.4, -0.4},
                                            {0.2, 0.2, -0.8},
                                            {0.4, 0.1, -0.5},
                                            {0.1, 0.1, -0.7},
                                            {0.2, 0.3, -0.9},
                                            {0.3, -0.2, -0.2},
                                            {0.1, 0.1, -0.6},
                                            {0.05, -0.4, -0.4},
                                            {0.1, 0.2, -0.7}},
                                           device);
        source_pcd.SetPointNormals(source_normals.To(device, dtype));

        t::pipelines::registration::TransformationEstimationSymmetric
                estimation;

        // Test RMSE computation
        double rmse = estimation.ComputeRMSE(source_pcd, target_pcd, corres);
        EXPECT_GT(rmse, 0.0);
        EXPECT_TRUE(std::isfinite(rmse));

        // Test transformation computation
        core::Tensor transform = estimation.ComputeTransformation(
                source_pcd, target_pcd, corres);
        EXPECT_EQ(transform.GetShape(), core::SizeVector({4, 4}));
        EXPECT_EQ(transform.GetDevice(), device);
        EXPECT_TRUE(transform.AllClose(transform));  // Check for NaN/Inf
    }
}

TEST_P(TransformationEstimationPermuteDevices,
       SymmetricICPCPUvsGPUConsistency) {
    core::Device device = GetParam();

    // Skip CUDA consistency test if device is CPU
    if (device.GetType() == core::Device::DeviceType::CPU) {
        GTEST_SKIP() << "Skipping CPU vs GPU consistency test for CPU device";
    }

#ifdef BUILD_CUDA_MODULE
    for (auto dtype : {core::Float32, core::Float64}) {
        // Create identical test data for both CPU and GPU
        auto [source_cpu, target_cpu, corres_cpu] =
                GetTestPointCloudsAndCorrespondences(dtype,
                                                     core::Device("CPU:0"));

        // Add normals to source
        core::Tensor source_normals_cpu =
                core::Tensor::Init<double>({{-0.1, -0.2, -0.8},
                                            {0.2, -0.1, -0.7},
                                            {0.05, -0.4, -0.6},
                                            {-0.01, -0.3, -0.5},
                                            {0.3, -0.1, -0.7},
                                            {0.06, -0.4, -0.4},
                                            {0.2, 0.2, -0.8},
                                            {0.4, 0.1, -0.5},
                                            {0.1, 0.1, -0.7},
                                            {0.2, 0.3, -0.9},
                                            {0.3, -0.2, -0.2},
                                            {0.1, 0.1, -0.6},
                                            {0.05, -0.4, -0.4},
                                            {0.1, 0.2, -0.7}},
                                           core::Device("CPU:0"));
        source_cpu.SetPointNormals(
                source_normals_cpu.To(core::Device("CPU:0"), dtype));

        // Copy to GPU
        auto source_gpu = source_cpu.To(device);
        auto target_gpu = target_cpu.To(device);
        auto corres_gpu = corres_cpu.To(device);

        // Test on both devices
        t::pipelines::registration::TransformationEstimationSymmetric
                estimation;

        // Compute on CPU
        double rmse_cpu =
                estimation.ComputeRMSE(source_cpu, target_cpu, corres_cpu);
        core::Tensor transform_cpu = estimation.ComputeTransformation(
                source_cpu, target_cpu, corres_cpu);

        // Compute on GPU
        double rmse_gpu =
                estimation.ComputeRMSE(source_gpu, target_gpu, corres_gpu);
        core::Tensor transform_gpu = estimation.ComputeTransformation(
                source_gpu, target_gpu, corres_gpu);

        // Compare results - they should be very close
        EXPECT_NEAR(rmse_cpu, rmse_gpu, 1e-6);
        EXPECT_TRUE(transform_cpu.AllClose(
                transform_gpu.To(core::Device("CPU:0")), 1e-6, 1e-6));

        utility::LogInfo("CPU RMSE: {}, GPU RMSE: {}", rmse_cpu, rmse_gpu);
    }
#else
    GTEST_SKIP() << "CUDA not available, skipping CPU vs GPU consistency test";
#endif
}

TEST_P(TransformationEstimationPermuteDevices, SymmetricICPRobustKernels) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_pcd(device), target_pcd(device);
        core::Tensor corres;
        std::tie(source_pcd, target_pcd, corres) =
                GetTestPointCloudsAndCorrespondences(dtype, device);

        // Add normals to source
        core::Tensor source_normals =
                core::Tensor::Init<double>({{-0.1, -0.2, -0.8},
                                            {0.2, -0.1, -0.7},
                                            {0.05, -0.4, -0.6},
                                            {-0.01, -0.3, -0.5},
                                            {0.3, -0.1, -0.7},
                                            {0.06, -0.4, -0.4},
                                            {0.2, 0.2, -0.8},
                                            {0.4, 0.1, -0.5},
                                            {0.1, 0.1, -0.7},
                                            {0.2, 0.3, -0.9},
                                            {0.3, -0.2, -0.2},
                                            {0.1, 0.1, -0.6},
                                            {0.05, -0.4, -0.4},
                                            {0.1, 0.2, -0.7}},
                                           device);
        source_pcd.SetPointNormals(source_normals.To(device, dtype));

        // Test different robust kernels
        std::vector<t::pipelines::registration::RobustKernel> kernels = {
                t::pipelines::registration::RobustKernel(
                        t::pipelines::registration::RobustKernelMethod::L2Loss,
                        1.0, 1.0),
                t::pipelines::registration::RobustKernel(
                        t::pipelines::registration::RobustKernelMethod::L1Loss,
                        1.0, 1.0),
                t::pipelines::registration::RobustKernel(
                        t::pipelines::registration::RobustKernelMethod::
                                HuberLoss,
                        1.0, 1.0)};

        for (const auto& kernel : kernels) {
            t::pipelines::registration::TransformationEstimationSymmetric
                    estimation(kernel);

            double rmse =
                    estimation.ComputeRMSE(source_pcd, target_pcd, corres);
            core::Tensor transform = estimation.ComputeTransformation(
                    source_pcd, target_pcd, corres);

            // All kernels should produce valid results
            EXPECT_GT(rmse, 0.0);
            EXPECT_TRUE(std::isfinite(rmse));
            EXPECT_EQ(transform.GetShape(), core::SizeVector({4, 4}));
            EXPECT_TRUE(transform.AllClose(transform));  // Check for NaN/Inf
        }
    }
}

}  // namespace tests
}  // namespace open3d
