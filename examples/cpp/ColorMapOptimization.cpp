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

#include <vector>

#include "open3d/Open3D.h"

using namespace open3d;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    ColorMapOptimization [data_dir]");
    // clang-format on
    utility::LogInfo("");
}

std::tuple<geometry::TriangleMesh,
           std::vector<geometry::RGBDImage>,
           camera::PinholeCameraTrajectory>
PrepareDataset(const std::string& data_path) {
    // Read RGBD images
    std::vector<std::string> depth_filenames, color_filenames;
    utility::filesystem::ListFilesInDirectoryWithExtension(
            data_path + "/depth/", "png", depth_filenames);
    utility::filesystem::ListFilesInDirectoryWithExtension(
            data_path + "/image/", "jpg", color_filenames);
    if (depth_filenames.size() != color_filenames.size()) {
        utility::LogError(
                "The number of depth images {} does not match the number of "
                "color images {}.",
                depth_filenames.size(), color_filenames.size());
    }
    std::sort(depth_filenames.begin(), depth_filenames.end());
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<geometry::RGBDImage> rgbd_images;
    for (size_t i = 0; i < depth_filenames.size(); i++) {
        utility::LogDebug("reading {}...", depth_filenames[i]);
        auto depth = io::CreateImageFromFile(depth_filenames[i]);
        utility::LogDebug("reading {}...", color_filenames[i]);
        auto color = io::CreateImageFromFile(color_filenames[i]);
        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color, *depth, 1000.0, 3.0, false);
        rgbd_images.push_back(*rgbd_image);
    }

    // Camera trajectory.
    camera::PinholeCameraTrajectory camera_trajectory;
    io::ReadPinholeCameraTrajectory(data_path + "/scene/key.log",
                                    camera_trajectory);

    // Mesh.
    geometry::TriangleMesh mesh;
    io::ReadTriangleMesh(data_path + "/scene/integrated.ply", mesh);

    return std::make_tuple(mesh, rgbd_images, camera_trajectory);
}

/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014
int main(int argc, char* argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    // Read dataset.
    geometry::TriangleMesh mesh;
    std::vector<geometry::RGBDImage> rgbd_images;
    camera::PinholeCameraTrajectory camera_trajectory;
    std::tie(mesh, rgbd_images, camera_trajectory) = PrepareDataset(argv[1]);

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
