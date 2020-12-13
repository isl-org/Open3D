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

#include "open3d/Open3D.h"

using namespace open3d;

void VisualizeRegistration(const open3d::t::geometry::PointCloud &source,
                           const open3d::t::geometry::PointCloud &target,
                           const core::Tensor &transformation) {
    auto source_transformed = source;
    source_transformed.Transform(transformation);

    std::shared_ptr<geometry::PointCloud> source_transformed_ptr =
            std::make_shared<geometry::PointCloud>(
                    source_transformed.ToLegacyPointCloud());
    std::shared_ptr<geometry::PointCloud> source_ptr =
            std::make_shared<geometry::PointCloud>(source.ToLegacyPointCloud());
    std::shared_ptr<geometry::PointCloud> target_ptr =
            std::make_shared<geometry::PointCloud>(target.ToLegacyPointCloud());

    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

void PrintHelp() {
    utility::LogInfo("Usage :");
    utility::LogInfo("  > TensorPointCloudTransform <src_file> <target_file>");
}

int main(int argc, char *argv[]) {
    // TODO: Add argument input options for users and developers
    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    // TODO: Take this input as arguments
    auto device = core::Device("CUDA:0");
    auto dtype = core::Dtype::Float32;

    // TODO: Look for a neat method to import data on device
    // t::io::ReadPointCloud, changes the device to CPU and DType to Float64
    t::geometry::PointCloud source(device);
    t::geometry::PointCloud source2(device);
    t::geometry::PointCloud target(device);

    t::io::ReadPointCloud(argv[1], source, {"auto", false, false, true});
    t::io::ReadPointCloud(argv[2], target, {"auto", false, false, true});

    core::Tensor source_points = source.GetPoints().To(dtype).Copy(device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);
    core::Tensor target_points = target.GetPoints().To(dtype).Copy(device);
    core::Tensor target_normals =
            target.GetPointNormals().To(dtype).Copy(device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);
    // TODO: Look for a way to make pointcloud data of Float32 (it's Float64)

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
    // if using Float64, change float to double in the following vector
    // std::vector<float> trans_init_vec{
    //         0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
    //         0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};
    // // Creating Tensor from manual transformation vector
    // core::Tensor init_trans(trans_init_vec, {4, 4}, dtype, device);

    // Case 2: [Identity Transformation / No Transformation]
    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);

    // VisualizeRegistration(source, target, init_trans);

    utility::LogInfo(" Input on {} Success", device.ToString());

    // Running the EstimateTransformation function in loop, to get an
    // averged out time estimate.
    utility::Timer eval_timer;
    double avg_ = 0.0;
    double max_ = 0.0;
    double min_ = 1000000.0;
    int itr = 10;

    double max_correspondence_dist = 0.02;

    t::pipelines::registration::RegistrationResult evaluation(init_trans);
    for (int i = 0; i < itr; i++) {
        eval_timer.Start();
        evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
                source_device, target_device, max_correspondence_dist,
                init_trans);
        eval_timer.Stop();
        auto time = eval_timer.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)itr;

    // Printing result, [Time averaged over (itr) iterations]
    utility::LogInfo(" [Manually Initialised Transformation] ");
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_points.GetShape().ToString(),
                     target_points.GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            evaluation.correspondence_set_.GetShape()[0],
            max_correspondence_dist);
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
    auto reg_p2p = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));

    // Printing result for ICP Point to Point
    utility::LogInfo(" [ICP: Point to Point] ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_points.GetShape().ToString(),
                     target_points.GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            reg_p2p.correspondence_set_.GetShape()[0], max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", reg_p2p.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", reg_p2p.inlier_rmse_);

    // auto transformation_point2point = reg_p2p.transformation_;
    // VisualizeRegistration(source, target, transformation_point2point);

    // ICP: Point to Plane
    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));

    // Printing result for ICP Point to Plane
    utility::LogInfo(" [ICP: Point to Plane] ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_points.GetShape().ToString(),
                     target_points.GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            reg_p2p.correspondence_set_.GetShape()[0], max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", reg_p2plane.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", reg_p2plane.inlier_rmse_);

    // auto transformation_point2plane = reg_p2plane.transformation_;
    // VisualizeRegistration(source2, target, transformation_point2plane);

    return 0;
}
