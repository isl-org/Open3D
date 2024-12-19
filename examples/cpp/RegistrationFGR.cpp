// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
PreprocessPointCloud(const char *file_name, const float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down,
            open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd, pcd_down, pcd_fpfh);
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
    utility::LogInfo("    > RegistrationFGR source_pcd target_pcd"
                     "[--voxel_size=0.05] [--distance_multiplier=1.5]"
                     "[--max_iterations=64] [--max_tuples=1000]"
                     );
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    float voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    float distance_multiplier = utility::GetProgramOptionAsDouble(
            argc, argv, "--distance_multiplier", 1.5);
    float distance_threshold = voxel_size * distance_multiplier;
    int max_iterations =
            utility::GetProgramOptionAsInt(argc, argv, "--max_iterations", 64);
    int max_tuples =
            utility::GetProgramOptionAsInt(argc, argv, "--max_tuples", 1000);

    // Prepare input
    std::shared_ptr<geometry::PointCloud> source, source_down, target,
            target_down;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_down, source_fpfh) =
            PreprocessPointCloud(argv[1], voxel_size);
    std::tie(target, target_down, target_fpfh) =
            PreprocessPointCloud(argv[2], voxel_size);

    pipelines::registration::RegistrationResult registration_result =
            pipelines::registration::
                    FastGlobalRegistrationBasedOnFeatureMatching(
                            *source_down, *target_down, *source_fpfh,
                            *target_fpfh,
                            pipelines::registration::
                                    FastGlobalRegistrationOption(
                                            /* decrease_mu =  */ 1.4, true,
                                            true, distance_threshold,
                                            max_iterations,
                                            /* tuple_scale =  */ 0.95,
                                            max_tuples));

    VisualizeRegistration(*source, *target,
                          registration_result.transformation_);

    return 0;
}
