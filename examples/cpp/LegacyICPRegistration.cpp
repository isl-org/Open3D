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

#include <iostream>
#include <memory>

#include "Eigen/Eigen"
#include "open3d/Open3D.h"

using namespace open3d;

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
    utility::LogInfo("Usage :");
    utility::LogInfo("  > LegacyPointCloudTransform <src_file> <target_file>");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    // Creating Tensor PointCloud Input from argument specified file
    std::shared_ptr<open3d::geometry::PointCloud> source =
            open3d::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<open3d::geometry::PointCloud> target =
            open3d::io::CreatePointCloudFromFile(argv[2]);

    // For matching result with the Tutorial test case,
    // Comment out Case: 1 below [init_trans], and uncomment the Case: 2
    // for a good initial guess transformation, for pointcloud input
    // ../examples/test_data/ICP/cloud_bin_0.pcd
    // ../examples/test_data/ICP/cloud_bin_1.pcd
    // Expected result from evaluate registion (using HybridSearch):
    //
    // [PointCloud Size]: Source Points: {198835, 3} Target Points: {137833, 3}
    // [Correspondences]: 34741
    // Fitness: 0.174723
    // Inlier RMSE: 0.0117711

    // Case 1: [Tutorial case]
    Eigen::Matrix4d init_trans;
    init_trans << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7, 0.487,
            0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    // Case 2: [Identity Transformation / No Transformation]
    //     Eigen::Matrix4d init_trans = Eigen::Matrix4d::Identity();

    double max_correspondence_dist = 0.03;
    source = source->VoxelDownSample(0.01);
    target = target->VoxelDownSample(0.01);

    VisualizeRegistration(*source, *target, init_trans);

    // Running the EstimateTransformation function in loop, to get an
    // averged out time estimate.
    utility::Timer eval_timer;
    double avg_ = 0.0;
    double max_ = 0.0;
    double min_ = 1000000.0;
    int itr = 10;
    open3d::pipelines::registration::RegistrationResult evaluation(init_trans);
    for (int i = 0; i < itr; i++) {
        eval_timer.Start();
        evaluation = open3d::pipelines::registration::EvaluateRegistration(
                *source, *target, max_correspondence_dist, init_trans);
        eval_timer.Stop();
        auto time = eval_timer.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)itr;

    // Printing result, [Time averaged over (itr) iterations]
    utility::LogInfo(" [Manually Initialised Transformation] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            evaluation.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", evaluation.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", evaluation.inlier_rmse_);
    utility::LogInfo("     [TIME]: (averaged out for {} iterations)", itr);
    utility::LogInfo("       Average: {}   Max: {}   Min: {} ", avg_, max_,
                     min_);

    // VisualizeRegistration(source, target, init_trans);

    // ICP ConvergenceCriteria for both Point To Point and Point To Plane:
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 30;

    // ICP: Point to Point
    utility::Timer icp_p2p_time;
    icp_p2p_time.Start();
    auto reg_p2p = open3d::pipelines::registration::RegistrationICP(
            *source, *target, max_correspondence_dist, init_trans,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2p_time.Stop();

    // Printing result for ICP Point to Point
    utility::LogInfo(" [ICP: Point to Point] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo(
            "   [Correspondences]: {}, [maximum corrspondence distance = {}] ",
            reg_p2p.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("     Fitness: {} ", reg_p2p.fitness_);
    utility::LogInfo("     Inlier RMSE: {} ", reg_p2p.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2p_time.GetDuration());
    utility::LogInfo("     [Tranformation Matrix]: ");
    std::cout << reg_p2p.transformation_ << std::endl;

    // auto transformation_point2point = reg_p2p.transformation_;
    VisualizeRegistration(*source, *target, reg_p2p.transformation_);

    // ICP: Point to Plane
    utility::Timer icp_p2plane_time;
    icp_p2plane_time.Start();
    auto reg_p2plane = open3d::pipelines::registration::RegistrationICP(
            *source, *target, max_correspondence_dist, init_trans,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2plane_time.Stop();
    // Printing result for ICP Point to Plane
    utility::LogInfo(" [ICP: Point to Plane] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo(
            "   [Correspondences]: {}, [maximum corrspondence distance = {}] ",
            reg_p2plane.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("     Fitness: {} ", reg_p2plane.fitness_);
    utility::LogInfo("     Inlier RMSE: {} ", reg_p2plane.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2plane_time.GetDuration());
    utility::LogInfo("     [Tranformation Matrix]: ");
    std::cout << reg_p2plane.transformation_ << std::endl;

    // auto transformation_point2plane = reg_p2plane.transformation_;
    VisualizeRegistration(*source, *target, reg_p2plane.transformation_);

    return 0;
}