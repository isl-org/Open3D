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

// This example tests ICP Registration pipeline on the given pointcloud.
// To make things simple, and support any pointcloud for testing, input only
// requires 1 pointcloud source in argument, and the example automatically
// creates a target source by transforming the pointcloud, and estimating
// normals. Adjust the voxel_downsample_factor and max_correspondence_dist
// according to the test pointcloud.
//
//
// To run this example from Open3D directory:
// ./build/bin/example/test-tICP [device] [path to source pointcloud]
// [device] : CPU:0 / CUDA:0 ...
// [example path to source pointcloud relative to Open3D dir]:
// examples/test_data/ICP/cloud_bin_0.pcd

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

// Parameters to adjust according to the test pointcloud.
double voxel_downsample_factor = 1.0;
double max_correspondence_dist = 0.2;

// ICP ConvergenceCriteria:
double relative_fitness = 1e-6;
double relative_rmse = 1e-6;
int max_iterations = 5;

int main(int argc, char *argv[]) {
    // Argument 1: Device: 'CPU:0' for CPU, 'CUDA:0' for GPU
    // Argument 2: Path to the test PointCloud

    auto device = core::Device(argv[1]);
    auto dtype = core::Dtype::Float32;

    // t::io::ReadPointCloud, changes the device to CPU and DType to Float64
    t::geometry::PointCloud target_;
    // t::geometry::PointCloud target(device);
    t::io::ReadPointCloud(argv[2], target_, {"auto", false, false, true});

    utility::LogInfo(" Input Successful ");

    // geometry::PointCloud legacy_s = source_.ToLegacyPointCloud();
    geometry::PointCloud legacy_t = target_.ToLegacyPointCloud();

    // legacy_s.VoxelDownSample(voxel_downsample_factor);
    legacy_t.VoxelDownSample(voxel_downsample_factor);
    utility::LogInfo(" Downsampling Successful ");

    legacy_t.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(), false);
    utility::LogInfo(" Normal Estimation Successful ");

    t::geometry::PointCloud source =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    t::geometry::PointCloud target =
            t::geometry::PointCloud::FromLegacyPointCloud(legacy_t);

    // Creating Tensor from manual transformation vector
    core::Tensor trans =
            core::Tensor::Init<float>({{0.862, 0.011, -0.507, 0.5},
                                       {-0.139, 0.967, -0.215, 0.7},
                                       {0.487, 0.255, 0.835, -1.4},
                                       {0.0, 0.0, 0.0, 1.0}},
                                      core::Device("CPU:0"));
    target = target.Transform(trans);
    utility::LogInfo(" Target transformation Successful ");

    core::Tensor source_points =
            source.GetPoints().To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);
    utility::LogInfo(" Creating Source Pointcloud on device Successful ");

    core::Tensor target_points =
            target.GetPoints().To(device, dtype, /*copy=*/true);
    core::Tensor target_normals =
            target.GetPointNormals().To(device, dtype, /*copy=*/true);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);
    utility::LogInfo(" Creating Target Pointcloud on device Successful ");

    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);

    utility::LogInfo(" Input Process on {} Success", device.ToString());

    t::pipelines::registration::RegistrationResult evaluation(trans);
    evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
            source_device, target_device, max_correspondence_dist, init_trans);
    utility::LogInfo(" EvaluateRegistration Success", device.ToString());

    // ICP: Point to Plane
    utility::Timer icp_p2plane_time;
    icp_p2plane_time.Start();
    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    icp_p2plane_time.Stop();
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
    utility::LogInfo("       Fitness: {} ", reg_p2plane.fitness_);
    utility::LogInfo("       Inlier RMSE: {} ", reg_p2plane.inlier_rmse_);
    utility::LogInfo("     [Time]: {}", icp_p2plane_time.GetDuration());
    utility::LogInfo("     [Transformation Matrix]: \n{}",
                     reg_p2plane.transformation_.ToString());

    return 0;
}
