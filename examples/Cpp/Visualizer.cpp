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

#include <iostream>
#include <memory>
#include <thread>

#include "Open3D/Open3D.h"

void PrintUsage() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Visualizer [mesh|spin|slowspin|pointcloud|rainbow|image|depth|editing|editmesh] [filename]");
    utility::LogInfo("    > Visualizer [animation] [filename] [trajectoryfile]");
    utility::LogInfo("    > Visualizer [rgbd] [color] [depth] [--rgbd_type]");
    // clang-format on
}
int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 3) {
        PrintUsage();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "mesh") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometries({mesh_ptr}, "Mesh", 1600, 900);
    } else if (option == "editmesh") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithVertexSelection(
                {mesh_ptr}, "Edit Mesh", 1600, 900);
    } else if (option == "spin") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithAnimationCallback(
                {mesh_ptr},
                [&](visualization::Visualizer *vis) {
                    vis->GetViewControl().Rotate(10, 0);
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    return false;
                },
                "Spin", 1600, 900);
    } else if (option == "slowspin") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        visualization::DrawGeometriesWithKeyCallbacks(
                {mesh_ptr},
                {{GLFW_KEY_SPACE,
                  [&](visualization::Visualizer *vis) {
                      vis->GetViewControl().Rotate(10, 0);
                      std::this_thread::sleep_for(
                              std::chrono::milliseconds(30));
                      return false;
                  }}},
                "Press Space key to spin", 1600, 900);
    } else if (option == "pointcloud") {
        auto cloud_ptr = std::make_shared<geometry::PointCloud>();
        if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        cloud_ptr->NormalizeNormals();
        visualization::DrawGeometries({cloud_ptr}, "PointCloud", 1600, 900);
    } else if (option == "rainbow") {
        auto cloud_ptr = std::make_shared<geometry::PointCloud>();
        if (io::ReadPointCloud(argv[2], *cloud_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        cloud_ptr->NormalizeNormals();
        cloud_ptr->colors_.resize(cloud_ptr->points_.size());
        double color_index = 0.0;
        double color_index_step = 0.05;

        auto update_colors_func = [&cloud_ptr](double index) {
            auto color_map_ptr = visualization::GetGlobalColorMap();
            for (auto &c : cloud_ptr->colors_) {
                c = color_map_ptr->GetColor(index);
            }
        };
        update_colors_func(1.0);

        visualization::DrawGeometriesWithAnimationCallback(
                {cloud_ptr},
                [&](visualization::Visualizer *vis) {
                    color_index += color_index_step;
                    if (color_index > 2.0) color_index -= 2.0;
                    update_colors_func(fabs(color_index - 1.0));
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    return true;
                },
                "Rainbow", 1600, 900);
    } else if (option == "image") {
        auto image_ptr = std::make_shared<geometry::Image>();
        if (io::ReadImage(argv[2], *image_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        visualization::DrawGeometries({image_ptr}, "Image", image_ptr->width_,
                                      image_ptr->height_);
    } else if (option == "rgbd") {
        if (argc < 4) {
            PrintUsage();
            return 1;
        }

        int rgbd_type =
                utility::GetProgramOptionAsInt(argc, argv, "--rgbd_type", 0);
        auto color_ptr = std::make_shared<geometry::Image>();
        auto depth_ptr = std::make_shared<geometry::Image>();

        if (io::ReadImage(argv[2], *color_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }

        if (io::ReadImage(argv[3], *depth_ptr)) {
            utility::LogInfo("Successfully read {}", argv[3]);
        } else {
            utility::LogWarning("Failed to read {}", argv[3]);
            return 1;
        }

        std::shared_ptr<geometry::RGBDImage> (*CreateRGBDImage)(
                const geometry::Image &, const geometry::Image &, bool);
        if (rgbd_type == 0)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        else if (rgbd_type == 1)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromTUMFormat;
        else if (rgbd_type == 2)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromSUNFormat;
        else if (rgbd_type == 3)
            CreateRGBDImage = &geometry::RGBDImage::CreateFromNYUFormat;
        else
            CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        auto rgbd_ptr = CreateRGBDImage(*color_ptr, *depth_ptr, false);
        visualization::DrawGeometries({rgbd_ptr}, "RGBD", depth_ptr->width_ * 2,
                                      depth_ptr->height_);

    } else if (option == "depth") {
        auto image_ptr = io::CreateImageFromFile(argv[2]);
        camera::PinholeCameraIntrinsic camera;
        camera.SetIntrinsics(640, 480, 575.0, 575.0, 319.5, 239.5);
        auto pointcloud_ptr =
                geometry::PointCloud::CreateFromDepthImage(*image_ptr, camera);
        visualization::DrawGeometries(
                {pointcloud_ptr},
                "geometry::PointCloud from Depth geometry::Image", 1920, 1080);
    } else if (option == "editing") {
        auto pcd = io::CreatePointCloudFromFile(argv[2]);
        visualization::DrawGeometriesWithEditing({pcd}, "Editing", 1920, 1080);
    } else if (option == "animation") {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(argv[2], *mesh_ptr)) {
            utility::LogInfo("Successfully read {}", argv[2]);
        } else {
            utility::LogWarning("Failed to read {}", argv[2]);
            return 1;
        }
        mesh_ptr->ComputeVertexNormals();
        if (argc == 3) {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1920, 1080);
        } else {
            visualization::DrawGeometriesWithCustomAnimation(
                    {mesh_ptr}, "Animation", 1600, 900, 50, 50, argv[3]);
        }
    }

    utility::LogInfo("End of the test.");

    return 0;
}
