// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Program that runs the tensor version of Doppler ICP registration.
//
// A sample sequence of Doppler ICP point clouds (in XYZD format) is provided in
// open3d.data.DemoDopplerICPSequence.
//
// This is the implementation of the following paper:
// B. Hexsel, H. Vhavle, Y. Chen,
// DICP: Doppler Iterative Closest Point Algorithm, RSS 2022.

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

namespace {

// Parameters for Doppler ICP registration.
constexpr double kLambdaDoppler{0.01};
constexpr bool kRejectDynamicOutliers{false};
constexpr double kDopplerOutlierThreshold{2.0};
constexpr std::size_t kOutlierRejectionMinIteration{2};
constexpr std::size_t kGeometricRobustLossMinIteration{0};
constexpr std::size_t kDopplerRobustLossMinIteration{2};
constexpr double kTukeyLossScale{0.5};
constexpr double kNormalsSearchRadius{10.0};
constexpr int kNormalsMaxNeighbors{30};
constexpr double kMaxCorrespondenceDist{0.3};
constexpr std::size_t kUniformDownsampleFactor{2};
constexpr double kFitnessEpsilon{1e-6};
constexpr std::size_t kMaxIters{200};

}  // namespace

void VisualizeRegistration(const geometry::PointCloud &source,
                           const geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

core::Tensor ComputeDirectionVectors(const core::Tensor &positions) {
    utility::LogDebug("ComputeDirectionVectors");

    core::Tensor directions = core::Tensor::Empty(
            positions.GetShape(), positions.GetDtype(), positions.GetDevice());

    for (int64_t i = 0; i < positions.GetLength(); ++i) {
        // Compute the norm of the position vector.
        core::Tensor norm = (positions[i][0] * positions[i][0] +
                             positions[i][1] * positions[i][1] +
                             positions[i][2] * positions[i][2])
                                    .Sqrt();

        // If the norm is zero, set the direction vector to zero.
        if (norm.Item<float>() == 0.0) {
            directions[i].Fill(0.0);
        } else {
            // Otherwise, compute the direction vector by dividing the position
            // vector by its norm.
            directions[i] = positions[i] / norm;
        }
    }

    return directions;
}

void PrintHelp() {
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RegistrationDopplerICP source_idx target_idx [--visualize --cuda]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    bool visualize = false;
    if (utility::ProgramOptionExists(argc, argv, "--visualize")) {
        visualize = true;
    }

    // Device.
    core::Device device("CPU:0");
    if (utility::ProgramOptionExistsAny(argc, argv, {"--cuda"})) {
        device = core::Device("CUDA:0");
    }
    utility::LogInfo("Selected device: {}", device.ToString());

    data::DemoDopplerICPSequence demo_sequence;

    t::geometry::PointCloud source;
    t::geometry::PointCloud target;

    // Read point clouds, t::io::ReadPointCloud copies the pointcloud to CPU.
    t::io::ReadPointCloud(demo_sequence.GetPath(std::stoi(argv[1])), source,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(demo_sequence.GetPath(std::stoi(argv[2])), target,
                          {"auto", false, false, true});

    // Set direction vectors.
    source.SetPointAttr("directions",
                        ComputeDirectionVectors(source.GetPointPositions()));

    source = source.To(device).UniformDownSample(kUniformDownsampleFactor);
    target = target.To(device).UniformDownSample(kUniformDownsampleFactor);

    target.EstimateNormals(kNormalsSearchRadius, kNormalsMaxNeighbors);

    Eigen::Matrix4d calibration{Eigen::Matrix4d::Identity()};
    double period{0.0};
    demo_sequence.GetCalibration(calibration, period);

    // Vehicle to sensor frame calibration (T_V_S).
    const core::Tensor calibration_vehicle_to_sensor =
            core::eigen_converter::EigenMatrixToTensor(calibration).To(device);

    // Initial transform from source to target, to initialize ICP.
    core::Tensor initial_transform =
            core::Tensor::Eye(4, core::Dtype::Float64, device);

    auto kernel = t::pipelines::registration::RobustKernel(
            t::pipelines::registration::RobustKernelMethod::TukeyLoss,
            kTukeyLossScale);

    t::pipelines::registration::RegistrationResult result =
            t::pipelines::registration::ICP(
                    source, target, kMaxCorrespondenceDist, initial_transform,
                    t::pipelines::registration::
                            TransformationEstimationForDopplerICP(
                                    period, kLambdaDoppler,
                                    kRejectDynamicOutliers,
                                    kDopplerOutlierThreshold,
                                    kOutlierRejectionMinIteration,
                                    kGeometricRobustLossMinIteration,
                                    kDopplerRobustLossMinIteration, kernel,
                                    kernel, calibration_vehicle_to_sensor),
                    t::pipelines::registration::ICPConvergenceCriteria(
                            kFitnessEpsilon, kFitnessEpsilon, kMaxIters));

    if (visualize) {
        VisualizeRegistration(source.ToLegacy(), target.ToLegacy(),
                              core::eigen_converter::TensorToEigenMatrixXd(
                                      result.transformation_));
    }

    // Get the ground truth pose.
    auto trajectory = demo_sequence.GetTrajectory();
    const core::Tensor pose_source = core::eigen_converter::EigenMatrixToTensor(
            trajectory[std::stoi(argv[1])].second);
    const core::Tensor pose_target = core::eigen_converter::EigenMatrixToTensor(
            trajectory[std::stoi(argv[2])].second);
    const core::Tensor target_to_source =
            pose_target.Inverse().Matmul(pose_source);

    utility::LogInfo(
            "Estimated pose [rx ry rz tx ty tz]: \n{}",
            t::pipelines::kernel::TransformationToPose(result.transformation_)
                    .ToString(false));

    utility::LogInfo(
            "Ground truth pose [rx ry rz tx ty tz]: \n{}",
            t::pipelines::kernel::TransformationToPose(target_to_source)
                    .ToString(false));

    return EXIT_SUCCESS;
}
