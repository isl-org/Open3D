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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
    using namespace open3d;
    PrintOpen3DVersion();
    PrintInfo("Usage:\n");
    PrintInfo("    > ViewGeometry [options]\n");
    PrintInfo("      Open a window to view geometry.\n");
    PrintInfo("\n");
    PrintInfo("Basic options:\n");
    PrintInfo("    --help, -h                : Print help information.\n");
    PrintInfo("    --mesh file               : Add a triangle mesh from file.\n");
    PrintInfo("    --pointcloud file         : Add a point cloud from file.\n");
    PrintInfo("    --image file              : Add an image from file.\n");
    PrintInfo("    --depth file              : Add a point cloud converted from a depth image.\n");
    PrintInfo("    --depth_camera file       : Use with --depth, read a json file that stores\n");
    PrintInfo("                                the camera parameters.\n");
    PrintInfo("    --show_frame              : Add a coordinate frame.\n");
    PrintInfo("    --verbose n               : Set verbose level (0-4).\n");
    PrintInfo("\n");
    PrintInfo("Animation options:\n");
    PrintInfo("    --render_option file      : Read a json file of rendering settings.\n");
    PrintInfo("    --view_trajectory file    : Read a json file of view trajectory.\n");
    PrintInfo("    --camera_trajectory file  : Read a json file of camera trajectory.\n");
    PrintInfo("    --auto_recording [i|d]    : Automatically plays the animation, record\n");
    PrintInfo("                                images (i) or depth images (d). Exits when\n");
    PrintInfo("                                animation ends.\n");
    PrintInfo("\n");
    PrintInfo("Window options:\n");
    PrintInfo("    --window_name name        : Set window name.\n");
    PrintInfo("    --height n                : Set window height.\n");
    PrintInfo("    --width n                 : Set window width.\n");
    PrintInfo("    --top n                   : Set window top edge.\n");
    PrintInfo("    --left n                  : Set window left edge.\n");
}

int main(int argc, char **argv)
{
    using namespace open3d;
    using namespace open3d::filesystem;

    int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    SetVerbosityLevel((VerbosityLevel)verbose);
    if (argc <= 1 || ProgramOptionExists(argc, argv, "--help") ||
            ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    std::vector<std::shared_ptr<Geometry>> geometry_ptrs;
    int width = GetProgramOptionAsInt(argc, argv, "--width", 1920);
    int height = GetProgramOptionAsInt(argc, argv, "--height", 1080);
    int top = GetProgramOptionAsInt(argc, argv, "--top", 200);
    int left = GetProgramOptionAsInt(argc, argv, "--left", 200);
    std::string window_name = GetProgramOptionAsString(argc, argv,
            "--window_name", "ViewGeometry");
    std::string mesh_filename = GetProgramOptionAsString(argc, argv, "--mesh");
    std::string pcd_filename = GetProgramOptionAsString(argc, argv,
            "--pointcloud");
    std::string image_filename = GetProgramOptionAsString(argc, argv,
            "--image");
    std::string depth_filename = GetProgramOptionAsString(argc, argv,
            "--depth");
    std::string depth_parameter_filename = GetProgramOptionAsString(argc, argv,
            "--depth_camera");
    std::string render_filename = GetProgramOptionAsString(argc, argv,
            "--render_option");
    std::string view_filename = GetProgramOptionAsString(argc, argv,
            "--view_trajectory");
    std::string camera_filename = GetProgramOptionAsString(argc, argv,
            "--camera_trajectory");
    bool show_coordinate_frame = ProgramOptionExists(argc, argv,
            "--show_frame");

    VisualizerWithCustomAnimation visualizer;
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left, top) ==
            false) {
        PrintWarning("Failed creating OpenGL window.\n");
        return 0;
    }

    if (!mesh_filename.empty()) {
        auto mesh_ptr = CreateMeshFromFile(mesh_filename);
        mesh_ptr->ComputeVertexNormals();
        if (visualizer.AddGeometry(mesh_ptr) == false) {
            PrintWarning("Failed adding triangle mesh.\n");
        }
    }
    if (!pcd_filename.empty()) {
        auto pointcloud_ptr = CreatePointCloudFromFile(pcd_filename);
        if (visualizer.AddGeometry(pointcloud_ptr) == false) {
            PrintWarning("Failed adding point cloud.\n");
        }
        if (pointcloud_ptr->points_.size() > 5000000) {
            visualizer.GetRenderOption().point_size_ = 1.0;
        }
    }
    if (!image_filename.empty()) {
        auto image_ptr = CreateImageFromFile(image_filename);
        if (visualizer.AddGeometry(image_ptr) == false) {
            PrintWarning("Failed adding image.\n");
        }
    }
    if (!depth_filename.empty()) {
        PinholeCameraTrajectory intrinsic;
        if (depth_parameter_filename.empty() ||
                !ReadIJsonConvertible(depth_parameter_filename, intrinsic)) {
            PrintWarning("Failed to read intrinsic parameters for depth image.\n");
            PrintWarning("Use default value for Primesense camera.\n");
            intrinsic.intrinsic_.SetIntrinsics(640, 480, 525.0, 525.0, 319.5,
                    239.5);
        }
        auto image_ptr = CreateImageFromFile(depth_filename);
        auto pointcloud_ptr = CreatePointCloudFromDepthImage(*image_ptr,
                intrinsic.intrinsic_, intrinsic.extrinsic_.empty() ?
                Eigen::Matrix4d::Identity() : intrinsic.extrinsic_[0]);
        if (visualizer.AddGeometry(pointcloud_ptr) == false) {
            PrintWarning("Failed adding depth image.\n");
        }
    }

    if (visualizer.HasGeometry() == false) {
        PrintWarning("No geometry to render!\n");
        visualizer.DestroyVisualizerWindow();
        return 0;
    }

    if (!render_filename.empty()) {
        if (ReadIJsonConvertible(render_filename,
                visualizer.GetRenderOption()) == false) {
            PrintWarning("Failed loading rendering settings.\n");
        }
    }

    if (!view_filename.empty()) {
        auto &view_control =
                (ViewControlWithCustomAnimation &)visualizer.GetViewControl();
        if (view_control.LoadTrajectoryFromJsonFile(view_filename) == false) {
            PrintWarning("Failed loading view trajectory.\n");
        }
    } else if (!camera_filename.empty()) {
        PinholeCameraTrajectory camera_trajectory;
        if (ReadIJsonConvertible(camera_filename, camera_trajectory) == false) {
            PrintWarning("Failed loading camera trajectory.\n");
        } else {
            auto &view_control = (ViewControlWithCustomAnimation &)
                    visualizer.GetViewControl();
            if (view_control.LoadTrajectoryFromCameraTrajectory(
                    camera_trajectory) == false) {
                PrintWarning("Failed converting camera trajectory to view trajectory.\n");
            }
        }
    }

    visualizer.GetRenderOption().show_coordinate_frame_ = show_coordinate_frame;

    if (ProgramOptionExists(argc, argv, "--auto_recording")) {
        std::string mode = GetProgramOptionAsString(argc, argv,
                "--auto_recording");
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

    return 1;
}
