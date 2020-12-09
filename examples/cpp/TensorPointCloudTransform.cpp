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

// Evaluate Registration, Transformation Operation and Visualisation
// for Tensor Point Cloud
// Usage:
// .[path]/TensorPointCloudTransform <path source pcd> <path target pcd>
// Example <Source pcd> <Target pcd>:
//      ../examples/test_data/ICP/cloud_bin_0.pcd
//      ../examples/test_data/ICP/cloud_bin_1.pcd

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/Registration.h"

using namespace open3d;

void PrintHelp() {
    utility::LogInfo("Usage :");
    utility::LogInfo("  > TensorPointCloudTransform <src_file> <target_file>");
}

void ManualTransformationVisualizeRegistration(
        const std::shared_ptr<open3d::t::geometry::PointCloud>
                &tsource_transformed_ptr,
        const std::shared_ptr<open3d::t::geometry::PointCloud> &ttarget_ptr) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = tsource_transformed_ptr->ToLegacyPointCloud();
    *target_ptr = ttarget_ptr->ToLegacyPointCloud();
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "TensorPointCloudTransform result");
}

int main(int argc, char *argv[]) {
    // TODO: Add argument input options for users and developers
    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    auto device = core::Device("CPU:0");
    auto dtype = core::Dtype::Float64;

    // Creating Tensor PointCloud Input from argument specified file
    std::shared_ptr<open3d::t::geometry::PointCloud> tsource =
            open3d::t::io::CreatetPointCloudFromFile(argv[1]);
    std::shared_ptr<open3d::t::geometry::PointCloud> ttarget =
            open3d::t::io::CreatetPointCloudFromFile(argv[2]);

    // Manual Transformation
    std::vector<double> trans_init_vec{
            0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
            0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};

    // Creating Tensor from manual transformation vector
    core::Tensor trans_init_tensor(trans_init_vec, {4, 4}, dtype, device);
    auto evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
            *tsource, *ttarget, 0.02, trans_init_tensor);

    std::cout << " [Tensor] Registration Results: " << std::endl
              << "   Fitness: " << evaluation.fitness_ << std::endl
              << "   Inlier RMSE: " << evaluation.inlier_rmse_ << std::endl;

    // Testing Transformation Function
    tsource->Transform(trans_init_tensor);
    // converts tensor pointclouod back to legacy for visualisation
    ManualTransformationVisualizeRegistration(tsource, ttarget);

    return 0;
}