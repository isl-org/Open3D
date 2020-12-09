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
// for Legacy Point Cloud
// Usage:
// .[path]/LegacyPointCloudTransform <path source pcd> <path target pcd>
// Example <Source pcd> <Target pcd>:
//      ../examples/test_data/ICP/cloud_bin_0.pcd
//      ../examples/test_data/ICP/cloud_bin_1.pcd

#include <iostream>
#include <memory>

#include "Eigen/Eigen"
#include "open3d/Open3D.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/Registration.h"

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
    // TODO: Add argument input options for users and developers
    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    // Creating Tensor PointCloud Input from argument specified file
    std::shared_ptr<open3d::geometry::PointCloud> source =
            open3d::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<open3d::geometry::PointCloud> target =
            open3d::io::CreatePointCloudFromFile(argv[2]);

    Eigen::Matrix4d trans_init_eigen;
    trans_init_eigen << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7,
            0.487, 0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;

    auto evaluation = open3d::pipelines::registration::EvaluateRegistration(
            *source, *target, 0.02, trans_init_eigen);

    std::cout << " [Legacy] Registration Results: " << std::endl
              << "   Fitness: " << evaluation.fitness_ << std::endl
              << "   Inlier RMSE: " << evaluation.inlier_rmse_ << std::endl;

    VisualizeRegistration(*source, *target, trans_init_eigen);

    return 0;
}