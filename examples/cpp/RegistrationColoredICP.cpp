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
    utility::LogInfo("    > RegistrationColoredICP source_pcd target_pcd [--visualize]");
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

    // Prepare input
    std::shared_ptr<geometry::PointCloud> source =
            open3d::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<geometry::PointCloud> target =
            open3d::io::CreatePointCloudFromFile(argv[2]);
    if (source == nullptr || target == nullptr) {
        utility::LogWarning("Unable to load source or target file.");
        return -1;
    }

    std::vector<double> voxel_sizes = {0.05, 0.05 / 2, 0.05 / 4};
    std::vector<int> iterations = {50, 30, 14};
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        float voxel_size = voxel_sizes[i];

        auto source_down = source->VoxelDownSample(voxel_size);
        source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto target_down = target->VoxelDownSample(voxel_size);
        target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto result = pipelines::registration::RegistrationICP(
                *source_down, *target_down, 0.07, trans,
                pipelines::registration::
                        TransformationEstimationForColoredICP(),
                pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6,
                                                                iterations[i]));
        trans = result.transformation_;

        if (visualize) {
            VisualizeRegistration(*source, *target, trans);
        }
    }

    std::stringstream ss;
    ss << trans;
    utility::LogInfo("Final transformation = \n{}", ss.str());

    return 0;
}
