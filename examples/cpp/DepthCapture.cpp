// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <thread>

#include "open3d/Open3D.h"

using namespace open3d;

class VisualizerWithDepthCapture
    : public visualization::VisualizerWithCustomAnimation {
protected:
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override {
        if (action == GLFW_RELEASE) {
            return;
        }
        if (key == GLFW_KEY_S) {
            CaptureDepthImage("depth.png");
            CaptureDepthPointCloud("depth.ply");
            camera::PinholeCameraTrajectory camera;
            camera.parameters_.resize(1);
            view_control_ptr_->ConvertToPinholeCameraParameters(
                    camera.parameters_[0]);
            io::WriteIJsonConvertible("camera.json", camera);
        } else if (key == GLFW_KEY_L) {
            if (utility::filesystem::FileExists("depth.png") &&
                utility::filesystem::FileExists("camera.json")) {
                camera::PinholeCameraTrajectory camera;
                io::ReadIJsonConvertible("camera.json", camera);
                auto image_ptr = io::CreateImageFromFile("depth.png");
                auto pointcloud_ptr =
                        geometry::PointCloud::CreateFromDepthImage(
                                *image_ptr, camera.parameters_[0].intrinsic_,
                                camera.parameters_[0].extrinsic_);
                AddGeometry(pointcloud_ptr);
            }
        } else if (key == GLFW_KEY_K) {
            if (utility::filesystem::FileExists("depth.ply")) {
                auto pointcloud_ptr = io::CreatePointCloudFromFile("depth.ply");
                AddGeometry(pointcloud_ptr);
            }
        } else if (key == GLFW_KEY_P) {
            if (utility::filesystem::FileExists("depth.png") &&
                utility::filesystem::FileExists("camera.json")) {
                camera::PinholeCameraTrajectory camera;
                io::ReadIJsonConvertible("camera.json", camera);
                view_control_ptr_->ConvertFromPinholeCameraParameters(
                        camera.parameters_[0]);
            }
        } else {
            visualization::VisualizerWithCustomAnimation::KeyPressCallback(
                    window, key, scancode, action, mods);
        }
        UpdateRender();
    }
};

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > DepthCapture");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc != 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    data::BunnyMesh bunny_mesh;
    auto mesh_ptr = io::CreateMeshFromFile(bunny_mesh.GetPath());
    mesh_ptr->ComputeVertexNormals();
    utility::LogInfo("Press S to capture a depth image.");
    VisualizerWithDepthCapture visualizer;
    visualizer.CreateVisualizerWindow("Depth Capture", 640, 480, 200, 200);
    visualizer.AddGeometry(mesh_ptr);
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();

    if (!utility::filesystem::FileExists("depth.png") ||
        !utility::filesystem::FileExists("camera.json")) {
        utility::LogInfo("Depth has not been captured.");
        return 1;
    }

    auto image_ptr = io::CreateImageFromFile("depth.png");
    visualization::DrawGeometries({image_ptr});

    camera::PinholeCameraTrajectory camera;
    io::ReadIJsonConvertible("camera.json", camera);
    auto pointcloud_ptr = geometry::PointCloud::CreateFromDepthImage(
            *image_ptr, camera.parameters_[0].intrinsic_,
            camera.parameters_[0].extrinsic_);
    VisualizerWithDepthCapture visualizer1;
    visualizer1.CreateVisualizerWindow("Depth Validation", 640, 480, 200, 200);
    visualizer1.AddGeometry(pointcloud_ptr);
    visualizer1.Run();
    visualizer1.DestroyVisualizerWindow();

    utility::LogInfo("Press L to validate the depth image.");
    utility::LogInfo("Press P to load the capturing camera pose.");
    VisualizerWithDepthCapture visualizer2;
    visualizer2.CreateVisualizerWindow("Depth Validation", 640, 480, 200, 200);
    visualizer2.AddGeometry(mesh_ptr);
    visualizer2.Run();
    visualizer2.DestroyVisualizerWindow();

    utility::LogInfo("End of the test.");

    return 0;
}
