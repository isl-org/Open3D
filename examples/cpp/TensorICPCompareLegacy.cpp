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
    utility::LogInfo(
            "  > TensorPointCloudTransform <device> <src_file> <target_file> "
            "<voxel_size> <bool_init_trans>");
    utility::LogInfo(
            "  > [Source PCD Path]: (example) "
            "examples/test_data/ICP/cloud_bin_0.pcd ");
    utility::LogInfo(
            "  > [Target PCD Path]: (example) "
            "examples/test_data/ICP/cloud_bin_1.pcd ");
    utility::LogInfo("  > Device Option: 'CPU:0' or 'CUDA:0' ");
    utility::LogInfo("  > voxel size: 0.1 - 0.0001 ");
    utility::LogInfo("  > bool_init_trans: 0 - Identity Transformation Guess ");
    utility::LogInfo(
            "  >                  1 - Good guess for the above mentioned "
            "example/ ");
}

int main(int argc, char *argv[]) {
    // TODO: Add argument input options for users and developers
    if (argc < 5) {
        PrintHelp();
        return 1;
    }
    // Argument 1: Device: 'CPU:0' for CPU, 'CUDA:0' for GPU
    // Argument 2: Path to Source PointCloud
    // Argument 3: Path to Target PointCloud

    auto device = core::Device(argv[1]);
    auto dtype = core::Dtype::Float32;

    // Creating Tensor PointCloud Input from argument specified file
    std::shared_ptr<open3d::geometry::PointCloud> source_l =
            open3d::io::CreatePointCloudFromFile(argv[2]);
    std::shared_ptr<open3d::geometry::PointCloud> target_l =
            open3d::io::CreatePointCloudFromFile(argv[3]);

    utility::LogInfo(" Input Success as Legacy PointCloud ");
    // Voxel Downsampling
    double voxel_size = (double)std::stod(argv[4]);
    auto source_l_down = *(source_l->VoxelDownSample(voxel_size));
    auto target_l_down = *(target_l->VoxelDownSample(voxel_size));
    utility::LogInfo(" Downsampled the PointCloud by Voxel size = {}",
                     voxel_size);

    // Downsampled Tensor PointCloud
    auto source_device = t::geometry::PointCloud::FromLegacyPointCloud(
            source_l_down, dtype, device);
    auto target_device = t::geometry::PointCloud::FromLegacyPointCloud(
            target_l_down, dtype, device);

    utility::LogInfo(" Downsampled Tensor PointCloud created from Legacy ");
    utility::LogInfo("     Device: {},   Dtype: {} ",
                     source_device.GetDevice().ToString(),
                     source_device.GetPoints().GetDtype().ToString());

    // --- Now, source_device and target_device are our new tensor pointcloud
    // --- on "device" with Dtype = Float32

    // --- Initial Transformation input
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

    bool switch_init_transformation = (bool)std::stoi(argv[5]);
    core::Tensor init_trans_t;

    // --- Initial Transformation input for Tensor
    if (switch_init_transformation) {
        // Case 1: [Tutorial case]
        // if using Float64, change float to double in the following vector
        std::vector<float> trans_init_vec{
                0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
                0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};
        // Creating Tensor from manual transformation vector
        core::Tensor init_trans(trans_init_vec, {4, 4}, dtype, device);
        init_trans_t = init_trans;
    } else {
        // Case 2: [Identity Transformation / No Transformation]
        init_trans_t = core::Tensor::Eye(4, dtype, device);
    }

    // --- Initial Transformation input for Legacy
    Eigen::Matrix4d init_trans_l;
    if (switch_init_transformation) {
        // Case 1: [Tutorial case]
        init_trans_l << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7,
                0.487, 0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;
    } else {
        // Case 2: [Identity Transformation / No Transformation]
        init_trans_l = Eigen::Matrix4d::Identity();
    }

    utility::LogInfo(" [Tensor] Input on {} Success \n", device.ToString());
    utility::LogInfo("--------------------------------------------\n");

    // Running the EstimateTransformation function in loop, to get an
    // averged out time estimate.
    utility::Timer eval_timer;
    double avg_ = 0.0;
    double max_ = 0.0;
    double min_ = 1000000.0;
    int itr = 10;
    double max_correspondence_dist = 0.02;

    // --- TENSOR: EvaluateRegistration [based on HybridSearch]

    t::pipelines::registration::RegistrationResult evaluation_t(init_trans_t);
    for (int i = 0; i < itr; i++) {
        eval_timer.Start();
        evaluation_t = open3d::t::pipelines::registration::EvaluateRegistration(
                source_device, target_device, max_correspondence_dist,
                init_trans_t);
        eval_timer.Stop();
        auto time = eval_timer.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)itr;

    // Printing result, [Time averaged over (itr) iterations]
    utility::LogInfo(" [Tensor: Manually Initialised Transformation] ");
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_device.GetPoints().GetShape().ToString(),
                     target_device.GetPoints().GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            evaluation_t.correspondence_set_.GetShape()[0],
            max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", evaluation_t.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", evaluation_t.inlier_rmse_);
    utility::LogInfo("     [TIME]: (averaged out for {} iterations)", itr);
    utility::LogInfo("       Average: {}   Max: {}   Min: {} \n", avg_, max_,
                     min_);

    // --- LEGACY: EvaluateRegistration [based on HybridSearch]

    open3d::pipelines::registration::RegistrationResult evaluation_l(
            init_trans_l);
    for (int i = 0; i < itr; i++) {
        eval_timer.Start();
        evaluation_l = open3d::pipelines::registration::EvaluateRegistration(
                source_l_down, target_l_down, max_correspondence_dist,
                init_trans_l);
        eval_timer.Stop();
        auto time = eval_timer.GetDuration();
        avg_ += time;
        max_ = std::max(max_, time);
        min_ = std::min(min_, time);
    }
    avg_ = avg_ / (double)itr;

    // Printing result, [Time averaged over (itr) iterations]
    utility::LogInfo(" [Legacy: Manually Initialised Transformation] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            evaluation_l.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", evaluation_l.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", evaluation_l.inlier_rmse_);
    utility::LogInfo("     [TIME]: (averaged out for {} iterations)", itr);
    utility::LogInfo("       Average: {}   Max: {}   Min: {} \n", avg_, max_,
                     min_);
    utility::LogInfo("--------------------------------------------\n", itr);

    // VisualizeRegistration(source, target, init_trans);

    // ICP ConvergenceCriteria for both Point To Point and Point To Plane:
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 5;

    // --- TENSOR: ICP Point To Point

    // ICP: Point to Point
    utility::Timer icp_p2p_time;
    icp_p2p_time.Start();
    auto reg_p2p = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans_t,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2p_time.Stop();

    // // NEW RMSE AFTER APPLYING THE TRANSFORMATION
    // auto reg_p2p_l_new_rmse =
    // open3d::t::pipelines::registration::EvaluateRegistration(source_device,
    //  target_target_devicel_down, max_correspondence_dist,
    //  reg_p2p.transformation_);

    // Printing result for ICP Point to Point
    utility::LogInfo(" [Tensor: ICP: Point to Point] ");
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_device.GetPoints().GetShape().ToString(),
                     target_device.GetPoints().GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            reg_p2p.correspondence_set_.GetShape()[0], max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", reg_p2p.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", reg_p2p.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2p_time.GetDuration());
    utility::LogInfo("     [Transformation Matrix]: \n{} \n",
                     reg_p2p.transformation_.ToString());

    // --- Legacy: ICP Point To Point
    // ICP: Point to Point
    utility::Timer icp_p2p_time_l;
    icp_p2p_time_l.Start();
    auto reg_p2p_l = open3d::pipelines::registration::RegistrationICP(
            source_l_down, target_l_down, max_correspondence_dist, init_trans_l,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2p_time_l.Stop();

    // // NEW RMSE AFTER APPLYING THE TRANSFORMATION
    // auto reg_p2p_l_new_rmse =
    // open3d::pipelines::registration::EvaluateRegistration(source_l_down,
    //  target_l_down, max_correspondence_dist, reg_p2p_l.transformation_);

    // Printing result for ICP Point to Point
    utility::LogInfo(" [Legacy: ICP: Point to Point] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo(
            "   [Correspondences]: {}, [maximum corrspondence distance = {}] ",
            reg_p2p_l.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("     Fitness: {} ", reg_p2p_l.fitness_);
    utility::LogInfo("     Inlier RMSE: {} ", reg_p2p_l.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2p_time_l.GetDuration());
    utility::LogInfo("     [Tranformation Matrix]: ");
    std::cout << reg_p2p_l.transformation_ << std::endl << std::endl;
    utility::LogInfo("--------------------------------------------\n", itr);

    // auto transformation_point2point = reg_p2p.transformation_;
    // VisualizeRegistration(source, target, transformation_point2point);

    // --- Tensor: ICP: Point to Plane

    // ICP: Point to Plane
    utility::Timer icp_p2plane_time;
    icp_p2plane_time.Start();
    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans_t,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2plane_time.Stop();
    // Printing result for ICP Point to Plane
    utility::LogInfo(" [Tensor: ICP: Point to Plane] ");
    utility::LogInfo("   EvaluateRegistration on {} Success ",
                     device.ToString());
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo("     [PointCloud Size]: ");
    utility::LogInfo("       Points: {} Target Points: {} ",
                     source_device.GetPoints().GetShape().ToString(),
                     target_device.GetPoints().GetShape().ToString());
    utility::LogInfo(
            "     [Correspondences]: {}, [maximum corrspondence distance = "
            "{}] ",
            reg_p2plane.correspondence_set_.GetShape()[0],
            max_correspondence_dist);
    utility::LogInfo("       Fitness: {} ", reg_p2plane.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", reg_p2plane.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2plane_time.GetDuration());
    utility::LogInfo("     [Transformation Matrix]: \n{} \n",
                     reg_p2plane.transformation_.ToString());

    // --- Legacy Point To Plane

    // ICP: Point to Plane
    utility::Timer icp_p2plane_time_l;
    icp_p2plane_time_l.Start();
    auto reg_p2plane_l = open3d::pipelines::registration::RegistrationICP(
            source_l_down, target_l_down, max_correspondence_dist, init_trans_l,
            open3d::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2plane_time_l.Stop();

    // // NEW RMSE AFTER APPLYING THE TRANSFORMATION
    // auto reg_p2plane_l_new_rmse =
    // open3d::pipelines::registration::EvaluateRegistration(source_l_down,
    //  target_l_down, max_correspondence_dist, reg_p2plane_l.transformation_);

    // Printing result for ICP Point to Plane
    utility::LogInfo(" [Legacy: ICP: Point to Plane] ");
    utility::LogInfo(
            "   EvaluateRegistration on [Legacy CPU Implementation] Success ");
    utility::LogInfo("   Convergence Criteria: ");
    utility::LogInfo(
            "   Relative Fitness: {}, Relative Fitness: {}, Max Iterations {}",
            relative_fitness, relative_rmse, max_iterations);
    utility::LogInfo(
            "   [Correspondences]: {}, [maximum corrspondence distance = {}] ",
            reg_p2plane_l.correspondence_set_.size(), max_correspondence_dist);
    utility::LogInfo("     Fitness: {} ", reg_p2plane_l.fitness_);
    utility::LogInfo("     Inlier RMSE: {} ", reg_p2plane_l.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2plane_time_l.GetDuration());
    utility::LogInfo("     [Tranformation Matrix]: ");
    std::cout << reg_p2plane_l.transformation_ << std::endl;
    utility::LogInfo("--------------------------------------------\n", itr);

    // auto transformation_point2plane = reg_p2plane.transformation_;

    // auto transformation_point2plane = reg_p2plane.transformation_;
    // VisualizeRegistration(source2, target, transformation_point2plane);
    return 0;
}
