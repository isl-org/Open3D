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

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ViewGeometry [options]");
    utility::LogInfo("      Open a window to view geometry.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --mesh file               : Add a triangle mesh from file.");
    utility::LogInfo("    --pointcloud file         : Add a point cloud from file.");
    utility::LogInfo("    --lineset file            : Add a line set from file.");
    utility::LogInfo("    --voxelgrid file          : Add a voxel grid from file.");
    utility::LogInfo("    --image file              : Add an image from file.");
    utility::LogInfo("    --depth file              : Add a point cloud converted from a depth image.");
    utility::LogInfo("    --depth_camera file       : Use with --depth, read a json file that stores");
    utility::LogInfo("                                the camera parameters.");
    utility::LogInfo("    --show_frame              : Add a coordinate frame.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).");
    utility::LogInfo("");
    utility::LogInfo("Animation options:");
    utility::LogInfo("    --render_option file      : Read a json file of rendering settings.");
    utility::LogInfo("    --view_trajectory file    : Read a json file of view trajectory.");
    utility::LogInfo("    --camera_trajectory file  : Read a json file of camera trajectory.");
    utility::LogInfo("    --auto_recording [i|d]    : Automatically plays the animation, record");
    utility::LogInfo("                                images (i) or depth images (d). Exits when");
    utility::LogInfo("                                animation ends.");
    utility::LogInfo("");
    utility::LogInfo("Window options:");
    utility::LogInfo("    --window_name name        : Set window name.");
    utility::LogInfo("    --height n                : Set window height.");
    utility::LogInfo("    --width n                 : Set window width.");
    utility::LogInfo("    --top n                   : Set window top edge.");
    utility::LogInfo("    --left n                  : Set window left edge.");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;

    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    if (argc <= 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    std::vector<std::shared_ptr<geometry::Geometry>> geometry_ptrs;
    int width = utility::GetProgramOptionAsInt(argc, argv, "--width", 1920);
    int height = utility::GetProgramOptionAsInt(argc, argv, "--height", 1080);
    int top = utility::GetProgramOptionAsInt(argc, argv, "--top", 200);
    int left = utility::GetProgramOptionAsInt(argc, argv, "--left", 200);
    std::string window_name = utility::GetProgramOptionAsString(
            argc, argv, "--window_name", "ViewGeometry");
    std::string mesh_filename =
            utility::GetProgramOptionAsString(argc, argv, "--mesh");
    std::string pcd_filename =
            utility::GetProgramOptionAsString(argc, argv, "--pointcloud");
    std::string lineset_filename =
            utility::GetProgramOptionAsString(argc, argv, "--lineset");
    std::string voxelgrid_filename =
            utility::GetProgramOptionAsString(argc, argv, "--voxelgrid");
    std::string image_filename =
            utility::GetProgramOptionAsString(argc, argv, "--image");
    std::string depth_filename =
            utility::GetProgramOptionAsString(argc, argv, "--depth");
    std::string depth_parameter_filename =
            utility::GetProgramOptionAsString(argc, argv, "--depth_camera");
    std::string render_filename =
            utility::GetProgramOptionAsString(argc, argv, "--render_option");
    std::string view_filename =
            utility::GetProgramOptionAsString(argc, argv, "--view_trajectory");
    std::string camera_filename = utility::GetProgramOptionAsString(
            argc, argv, "--camera_trajectory");
    bool show_coordinate_frame =
            utility::ProgramOptionExists(argc, argv, "--show_frame");

    visualization::VisualizerWithCustomAnimation visualizer;
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogWarning("Failed creating OpenGL window.");
        return 0;
    }

    if (!mesh_filename.empty()) {
        auto mesh_ptr = io::CreateMeshFromFile(mesh_filename);
        mesh_ptr->ComputeVertexNormals();
        if (visualizer.AddGeometry(mesh_ptr) == false) {
            utility::LogWarning("Failed adding triangle mesh.");
            return 1;
        }
    }
    if (!pcd_filename.empty()) {
        auto pointcloud_ptr = io::CreatePointCloudFromFile(pcd_filename);
        if (visualizer.AddGeometry(pointcloud_ptr) == false) {
            utility::LogWarning("Failed adding point cloud.");
            return 1;
        }
        if (pointcloud_ptr->points_.size() > 5000000) {
            visualizer.GetRenderOption().point_size_ = 1.0;
        }
    }
    if (!lineset_filename.empty()) {
        auto lineset_ptr = io::CreateLineSetFromFile(lineset_filename);
        if (visualizer.AddGeometry(lineset_ptr) == false) {
            utility::LogWarning("Failed adding line set.");
            return 1;
        }
    }
    if (!voxelgrid_filename.empty()) {
        auto voxelgrid_ptr = io::CreateVoxelGridFromFile(voxelgrid_filename);
        if (visualizer.AddGeometry(voxelgrid_ptr) == false) {
            utility::LogWarning("Failed adding voxel grid.");
            return 1;
        }
    }
    if (!image_filename.empty()) {
        auto image_ptr = io::CreateImageFromFile(image_filename);
        if (visualizer.AddGeometry(image_ptr) == false) {
            utility::LogWarning("Failed adding image.");
            return 1;
        }
    }
    if (!depth_filename.empty()) {
        camera::PinholeCameraParameters parameters;
        if (depth_parameter_filename.empty() ||
            !io::ReadIJsonConvertible(depth_parameter_filename, parameters)) {
            utility::LogWarning(
                    "Failed to read intrinsic parameters for depth image.");
            utility::LogWarning("Using default value for Primesense camera.");
            parameters.intrinsic_.SetIntrinsics(640, 480, 525.0, 525.0, 319.5,
                                                239.5);
        }
        auto image_ptr = io::CreateImageFromFile(depth_filename);
        auto pointcloud_ptr = geometry::PointCloud::CreateFromDepthImage(
                *image_ptr, parameters.intrinsic_, parameters.extrinsic_);
        if (visualizer.AddGeometry(pointcloud_ptr) == false) {
            utility::LogWarning("Failed adding depth image.");
            return 1;
        }
    }

    if (visualizer.HasGeometry() == false) {
        utility::LogWarning("No geometry to render!");
        visualizer.DestroyVisualizerWindow();
        return 1;
    }

    if (!render_filename.empty()) {
        if (io::ReadIJsonConvertible(render_filename,
                                     visualizer.GetRenderOption()) == false) {
            utility::LogWarning("Failed loading rendering settings.");
            return 1;
        }
    }

    if (!view_filename.empty()) {
        auto &view_control = (visualization::ViewControlWithCustomAnimation &)
                                     visualizer.GetViewControl();
        if (view_control.LoadTrajectoryFromJsonFile(view_filename) == false) {
            utility::LogWarning("Failed loading view trajectory.");
            return 1;
        }
    } else if (!camera_filename.empty()) {
        camera::PinholeCameraTrajectory camera_trajectory;
        if (io::ReadIJsonConvertible(camera_filename, camera_trajectory) ==
            false) {
            utility::LogWarning("Failed loading camera trajectory.");
            return 1;
        } else {
            auto &view_control =
                    (visualization::ViewControlWithCustomAnimation &)
                            visualizer.GetViewControl();
            if (view_control.LoadTrajectoryFromCameraTrajectory(
                        camera_trajectory) == false) {
                utility::LogWarning(
                        "Failed converting camera trajectory to view "
                        "trajectory.");
                return 1;
            }
        }
    }

    visualizer.GetRenderOption().show_coordinate_frame_ = show_coordinate_frame;

    if (utility::ProgramOptionExists(argc, argv, "--auto_recording")) {
        std::string mode = utility::GetProgramOptionAsString(
                argc, argv, "--auto_recording");
        if (mode == "i") {
            visualizer.Play(true, false, true);
        } else if (mode == "d") {
            visualizer.Play(true, true, true);
        } else {
            visualizer.Play(true, false, true);
        }
        visualizer.Run();
    } else {
        visualizer.Run();
    }
    visualizer.DestroyVisualizerWindow();

    return 0;
}
