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

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::pipelines::registration;

std::vector<Eigen::Vector3d> ComputeDirectionVectors(
        const geometry::PointCloud &pcd) {
    utility::LogDebug("ComputeDirectionVectors");

    std::vector<Eigen::Vector3d> directions;
    size_t n_points = pcd.points_.size();
    directions.resize(n_points, Eigen::Vector3d::Zero());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)pcd.points_.size(); ++i) {
        directions[i] = pcd.points_[i].normalized();
    }

    return directions;
}

void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
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

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RegistrationDopplerICP source_pcd target_pcd [--visualize]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

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

    // Prepare input.
    std::shared_ptr<geometry::PointCloud> source =
            open3d::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<geometry::PointCloud> target =
            open3d::io::CreatePointCloudFromFile(argv[2]);
    if (source == nullptr || target == nullptr) {
        utility::LogWarning("Unable to load source or target file.");
        return -1;
    }

    // Configure DICP parameters.
    const double max_neighbor_distance = {0.3};
    const double lambda_doppler = {0.01};
    const bool reject_dynamic_outliers = {false};
    const double doppler_outlier_threshold = {2.0};
    const size_t outlier_rejection_min_iteration = {2};
    const size_t geometric_robust_loss_min_iteration = {0};
    const size_t doppler_robust_loss_min_iteration = {2};
    const double period = {0.1};  // seconds
    const double convergence_threshold = {1e-6};
    const size_t max_iters = {200};

    std::shared_ptr<RobustKernel> geometric_kernel =
            std::make_shared<TukeyLoss>(0.5);
    std::shared_ptr<RobustKernel> doppler_kernel =
            std::make_shared<TukeyLoss>(0.5);

    std::vector<Eigen::Vector3d> source_directions =
            ComputeDirectionVectors(*source);

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    auto result = RegistrationDopplerICP(
            *source, *target, source_directions, max_neighbor_distance,
            transform,
            TransformationEstimationForDopplerICP(
                    lambda_doppler, reject_dynamic_outliers,
                    doppler_outlier_threshold, outlier_rejection_min_iteration,
                    geometric_robust_loss_min_iteration,
                    doppler_robust_loss_min_iteration, geometric_kernel,
                    doppler_kernel),
            ICPConvergenceCriteria(convergence_threshold, convergence_threshold,
                                   max_iters),
            period, transform);
    transform = result.transformation_;

    std::stringstream ss;
    ss << transform;
    utility::LogInfo("Final transformation = \n{}", ss.str());

    if (visualize) {
        VisualizeRegistration(*source, *target, transform);
    }

    return 0;
}
