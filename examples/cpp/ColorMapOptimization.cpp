// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/Open3D.h"

using namespace open3d;

std::tuple<geometry::TriangleMesh,
           std::vector<geometry::RGBDImage>,
           camera::PinholeCameraTrajectory>
PrepareDataset() {
    data::SampleFountainRGBDImages fountain_rgbd;
    // Read RGBD images
    std::vector<geometry::RGBDImage> rgbd_images;
    for (size_t i = 0; i < fountain_rgbd.GetDepthPaths().size(); i++) {
        utility::LogDebug("reading {}...", fountain_rgbd.GetDepthPaths()[i]);
        auto depth = io::CreateImageFromFile(fountain_rgbd.GetDepthPaths()[i]);
        utility::LogDebug("reading {}...", fountain_rgbd.GetColorPaths()[i]);
        auto color = io::CreateImageFromFile(fountain_rgbd.GetColorPaths()[i]);
        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color, *depth, 1000.0, 3.0, false);
        rgbd_images.push_back(*rgbd_image);
    }

    // Camera trajectory.
    camera::PinholeCameraTrajectory camera_trajectory;
    io::ReadPinholeCameraTrajectory(fountain_rgbd.GetKeyframePosesLogPath(),
                                    camera_trajectory);

    // Mesh.
    geometry::TriangleMesh mesh;
    io::ReadTriangleMesh(fountain_rgbd.GetReconstructionPath(), mesh);

    return std::make_tuple(mesh, rgbd_images, camera_trajectory);
}

/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014
int main(int argc, char* argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // Read dataset.
    geometry::TriangleMesh mesh;
    std::vector<geometry::RGBDImage> rgbd_images;
    camera::PinholeCameraTrajectory camera_trajectory;
    std::tie(mesh, rgbd_images, camera_trajectory) = PrepareDataset();

    // Save averaged color map (iteration=0).
    pipelines::color_map::RigidOptimizerOption rigid_opt_option;
    rigid_opt_option.maximum_iteration_ = 0;
    std::tie(mesh, camera_trajectory) = pipelines::color_map::RunRigidOptimizer(
            mesh, rgbd_images, camera_trajectory, rigid_opt_option);
    io::WriteTriangleMesh("color_map_init.ply", mesh);

    // Run rigid optimization for 300 iterations.
    rigid_opt_option.maximum_iteration_ = 300;
    std::tie(mesh, camera_trajectory) = pipelines::color_map::RunRigidOptimizer(
            mesh, rgbd_images, camera_trajectory, rigid_opt_option);
    io::WriteTriangleMesh("color_map_rigid_opt.ply", mesh);

    // Run non-rigid optimization for 300 iterations.
    pipelines::color_map::NonRigidOptimizerOption non_rigid_option;
    non_rigid_option.maximum_iteration_ = 300;
    std::tie(mesh, camera_trajectory) =
            pipelines::color_map::RunNonRigidOptimizer(
                    mesh, rgbd_images, camera_trajectory, non_rigid_option);
    io::WriteTriangleMesh("color_map_non_rigid_opt.ply", mesh);

    return 0;
}
